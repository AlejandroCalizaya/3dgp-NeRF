#pragma once
// TCP client for the NeRF server (nerf/server.py). A worker thread always requests the latest frame
// for the current camera, so the UI stays smooth even if the NeRF is slow to render.
// It also lets the user choose which trained scene the server loads (protocol in server.py).
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class RemoteClient
{
public:
    ~RemoteClient() { stop(); }

    // Scenes offered by the server and the active one
    std::vector<std::string> scenes;
    int currentScene = 0;

    // Initial pose of the active scene (from the server)
    float initC2w[16];
    float target[3];
    float fovY = 0.8f;

    bool connectTo(const std::string &host, int port)
    {
        scenes.clear();
        pendingScene_ = -1;
        sceneChanged_ = false;
        fresh_ = false;
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
            return false;
        fd_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        // Timeouts during connect and the welcome message: a down or busy server must not freeze the window
        timeval tv{3, 0};
        if (fd_ >= 0)
        {
            setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv); // on Linux this also limits connect()
            setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        }
        if (fd_ < 0 || connect(fd_, res->ai_addr, res->ai_addrlen) != 0)
        {
            freeaddrinfo(res);
            if (fd_ >= 0)
                close(fd_);
            fd_ = -1;
            return false;
        }
        freeaddrinfo(res);
        int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

        // Welcome: scene list + current scene + initial pose
        int32_t n = 0;
        bool ok = recvAll(&n, 4) && n > 0 && n < 1000;
        for (int i = 0; ok && i < n; ++i)
        {
            char name[33] = {0};
            ok = recvAll(name, 32);
            scenes.emplace_back(name);
        }
        int32_t cur = 0;
        ok = ok && recvAll(&cur, 4) && readPose();
        if (!ok)
        {
            close(fd_);
            fd_ = -1;
            return false;
        }
        tv = {0, 0}; // the frame thread waits without limit (the server takes a while to load scenes)
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        currentScene = cur;
        running_ = true;
        worker_ = std::thread(&RemoteClient::loop, this);
        return true;
    }

    bool connected() const { return running_; }
    bool loadingScene() const { return pendingScene_ >= 0; }

    // Ask to change scene (the server takes a few seconds to load it)
    void selectScene(int idx) { pendingScene_ = idx; }

    // If the scene changed, returns true only once, with the new scene's initial pose
    bool takeSceneChange(float c2w[16], float tgt[3], float &fov)
    {
        std::lock_guard<std::mutex> lk(m_);
        if (!sceneChanged_)
            return false;
        std::memcpy(c2w, initC2w, 64);
        std::memcpy(tgt, target, 12);
        fov = fovY;
        sceneChanged_ = false;
        return true;
    }

    // Request a frame for this camera (c2w in column-major order, like glm)
    void request(int w, int h, float focal, const float *c2w)
    {
        std::lock_guard<std::mutex> lk(m_);
        req_.w = w;
        req_.h = h;
        req_.focal = focal;
        std::memcpy(req_.c2w, c2w, 64);
        req_.seq++;
    }

    // Copy the latest received frame if it is new. Returns true if there is a new image.
    bool fetch(std::vector<unsigned char> &out, int &w, int &h)
    {
        std::lock_guard<std::mutex> lk(m_);
        if (!fresh_)
            return false;
        out.swap(frame_);
        w = frameW_;
        h = frameH_;
        fresh_ = false;
        return true;
    }

    void stop()
    {
        running_ = false;
        if (fd_ >= 0)
            shutdown(fd_, SHUT_RDWR);
        if (worker_.joinable())
            worker_.join();
        if (fd_ >= 0)
            close(fd_);
        fd_ = -1;
    }

private:
    struct Req
    {
        int w = 0, h = 0;
        float focal = 0;
        float c2w[16];
        unsigned long seq = 0;
    };

    bool readPose()
    {
        float hello[20];
        if (!recvAll(hello, sizeof hello))
            return false;
        std::lock_guard<std::mutex> lk(m_);
        std::memcpy(initC2w, hello, 16 * sizeof(float));
        std::memcpy(target, hello + 16, 3 * sizeof(float));
        fovY = hello[19];
        return true;
    }

    bool recvAll(void *buf, size_t n)
    {
        auto *p = static_cast<char *>(buf);
        while (n > 0)
        {
            ssize_t r = recv(fd_, p, n, 0);
            if (r <= 0)
                return false;
            p += r;
            n -= r;
        }
        return true;
    }

    bool sendAll(const void *buf, size_t n)
    {
        auto *p = static_cast<const char *>(buf);
        while (n > 0)
        {
            ssize_t r = send(fd_, p, n, MSG_NOSIGNAL);
            if (r <= 0)
                return false;
            p += r;
            n -= r;
        }
        return true;
    }

    void loop()
    {
        unsigned long lastSeq = 0;
        std::vector<unsigned char> buf;
        while (running_)
        {
            // Pending scene change
            const int idx = pendingScene_;
            if (idx >= 0)
            {
                int32_t msg[2] = {1, idx};
                if (!sendAll(msg, sizeof msg) || !readPose())
                {
                    running_ = false;
                    break;
                }
                {
                    std::lock_guard<std::mutex> lk(m_);
                    currentScene = idx;
                    sceneChanged_ = true;
                    fresh_ = false;
                }
                pendingScene_ = -1;
                continue;
            }

            Req r;
            {
                std::lock_guard<std::mutex> lk(m_);
                r = req_;
            }
            if (r.seq == lastSeq || r.w <= 0)
            {
                usleep(2000);
                continue;
            }
            lastSeq = r.seq;
            char msg[4 + 12 + 64];
            const int32_t kind = 0;
            std::memcpy(msg, &kind, 4);
            std::memcpy(msg + 4, &r.w, 4);
            std::memcpy(msg + 8, &r.h, 4);
            std::memcpy(msg + 12, &r.focal, 4);
            std::memcpy(msg + 16, r.c2w, 64);
            buf.resize(static_cast<size_t>(r.w) * r.h * 3);
            if (!sendAll(msg, sizeof msg) || !recvAll(buf.data(), buf.size()))
            {
                running_ = false;
                break;
            }
            std::lock_guard<std::mutex> lk(m_);
            if (sceneChanged_)
                continue; // frame from the previous scene: discard
            frame_.swap(buf);
            frameW_ = r.w;
            frameH_ = r.h;
            fresh_ = true;
        }
    }

    int fd_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<int> pendingScene_{-1};
    std::thread worker_;
    std::mutex m_;
    Req req_;
    std::vector<unsigned char> frame_;
    int frameW_ = 0, frameH_ = 0;
    bool fresh_ = false;
    bool sceneChanged_ = false;
};
