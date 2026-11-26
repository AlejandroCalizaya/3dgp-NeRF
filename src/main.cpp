// NeRF viewer - graphics side: OpenGL window (GLFW + GLAD + ImGui).
// Shows an image (texture) full screen and lets the user move the camera.
// The image comes from the NeRF server, or from a placeholder renderer when no server is available.
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Camera.h"
#include "Renderer.h"
#include "RemoteClient.h"

#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char *kVert = R"(#version 330 core
out vec2 uv;
void main() {
    // Triangle covering the whole screen, no buffers
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
})";

static const char *kFrag = R"(#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D img;
void main() { frag = vec4(texture(img, uv).rgb, 1.0); })";

static unsigned int compile(GLenum type, const char *src)
{
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::cerr << "Shader error: " << log << "\n";
    }
    return s;
}

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Could not initialize GLFW.\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1024, 640, "NeRF Viewer", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Could not create the window.\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Could not initialize GLAD.\n";
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Program, empty VAO and image texture
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, compile(GL_VERTEX_SHADER, kVert));
    glAttachShader(prog, compile(GL_FRAGMENT_SHADER, kFrag));
    glLinkProgram(prog);
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    OrbitCamera camera;
    std::vector<unsigned char> image;
    int renderH = 240; // height of the rendered image (the width follows the window)
    int texW = 0, texH = 0;

    // NeRF server (nerf/server.py). If it is not available the placeholder renderer is used.
    RemoteClient remote;
    const char *hostEnv = std::getenv("NERF_HOST");
    const char *portEnv = std::getenv("NERF_PORT");
    const std::string host = hostEnv ? hostEnv : "host.docker.internal";
    const int port = portEnv ? std::atoi(portEnv) : 5555;
    if (remote.connectTo(host, port))
    {
        glm::mat4 c0 = glm::make_mat4(remote.initC2w);
        camera.setFromPose(c0, glm::make_vec3(remote.target), remote.fovY);
        std::cout << "Connected to NeRF server " << host << ":" << port << "\n";
    }
    else
        std::cout << "No NeRF server at " << host << ":" << port << " -> placeholder scene\n";
    OrbitCamera initialCamera = camera;
    int selectedScene = remote.currentScene;
    glm::mat4 lastPose(0.0f);
    double lastMove = 0.0;
    double lastTry = glfwGetTime();
    double lastX = 0, lastY = 0, lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const double now = glfwGetTime();
        const float dt = static_cast<float>(now - lastTime);
        lastTime = now;

        // --- Camera input ---
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (!io.WantCaptureMouse)
        {
            const float dx = static_cast<float>(x - lastX), dy = static_cast<float>(y - lastY);
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
                camera.orbit(dx, dy);
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
                glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
                camera.pan(dx, dy);
            if (io.MouseWheel != 0.0f)
                camera.zoom(io.MouseWheel);
        }
        lastX = x;
        lastY = y;
        if (!io.WantCaptureKeyboard)
        {
            const float v = 2.0f * dt;
            auto key = [&](int k) { return glfwGetKey(window, k) == GLFW_PRESS ? 1.0f : 0.0f; };
            camera.move(v * (key(GLFW_KEY_W) - key(GLFW_KEY_S)), v * (key(GLFW_KEY_D) - key(GLFW_KEY_A)),
                        v * (key(GLFW_KEY_E) - key(GLFW_KEY_Q)));
            if (key(GLFW_KEY_ESCAPE) > 0.0f)
                glfwSetWindowShouldClose(window, true);
        }

        // --- Render to the texture ---
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        // No server (never started or closed): retry every 2 s, so the start order does not matter
        if (!remote.connected() && now - lastTry > 2.0)
        {
            lastTry = now;
            remote.stop();
            if (remote.connectTo(host, port))
            {
                camera.setFromPose(glm::make_mat4(remote.initC2w), glm::make_vec3(remote.target), remote.fovY);
                initialCamera = camera;
                selectedScene = remote.currentScene;
                std::cout << "Connected to NeRF server " << host << ":" << port << " (" << remote.scenes.size()
                          << " scenes)\n";
            }
        }

        // New scene loaded by the server: reset the camera to its pose
        {
            float c2wNew[16], tgtNew[3], fovNew;
            if (remote.connected() && remote.takeSceneChange(c2wNew, tgtNew, fovNew))
            {
                camera.setFromPose(glm::make_mat4(c2wNew), glm::make_vec3(tgtNew), fovNew);
                initialCamera = camera;
            }
        }

        // While the camera moves a lower resolution is requested (NeRF is slow to render)
        const glm::mat4 curPose = camera.c2w();
        if (curPose != lastPose)
        {
            lastPose = curPose;
            lastMove = now;
        }
        const bool moving = remote.connected() && (now - lastMove) < 0.3;
        const int H = moving ? std::max(static_cast<int>(renderH * 0.4f), 64) : renderH;
        const int W = std::max(static_cast<int>(H * static_cast<float>(fbW) / std::max(fbH, 1)), 16);
        bool newImage = false;
        int W2 = W, H2 = H;
        if (remote.connected())
        {
            const glm::mat4 c = camera.c2w();
            remote.request(W, H, camera.focal(H), glm::value_ptr(c));
            newImage = remote.fetch(image, W2, H2);
        }
        else
        {
            renderPlaceholder(camera.c2w(), W, H, camera.focal(H), image);
            newImage = true;
        }
        if (newImage)
        {
            glBindTexture(GL_TEXTURE_2D, tex);
            if (W2 != texW || H2 != texH)
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, W2, H2, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
                texW = W2;
                texH = H2;
            }
            else
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W2, H2, GL_RGB, GL_UNSIGNED_BYTE, image.data());
        }

        // --- Drawing ---
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // --- UI ---
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera");
        ImGui::Text("%.1f FPS  |  render %dx%d  |  %s", io.Framerate, W, H,
                    remote.connected() ? "NeRF (server)" : "placeholder");
        if (!remote.connected())
            ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Looking for NeRF server at %s:%d ... (run 'make server')",
                               host.c_str(), port);
        if (remote.connected())
        {
            const char *preview = remote.scenes[selectedScene].c_str();
            if (ImGui::BeginCombo("Scene", preview))
            {
                for (int i = 0; i < static_cast<int>(remote.scenes.size()); ++i)
                    if (ImGui::Selectable(remote.scenes[i].c_str(), i == selectedScene) && i != selectedScene)
                    {
                        selectedScene = i;
                        remote.selectScene(i);
                    }
                ImGui::EndCombo();
            }
            if (remote.loadingScene())
                ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Loading scene...");
        }
        ImGui::SliderInt("Render height", &renderH, 64, 720);
        float fovDeg = glm::degrees(camera.fovY);
        if (ImGui::SliderFloat("FOV", &fovDeg, 20.0f, 100.0f))
            camera.fovY = glm::radians(fovDeg);
        if (ImGui::Button("Reset camera"))
            camera = initialCamera;
        ImGui::Separator();
        ImGui::Text("Camera-to-world pose (c2w):");
        const glm::mat4 c = camera.c2w();
        for (int r = 0; r < 3; ++r)
            ImGui::Text("%7.3f %7.3f %7.3f | %7.3f", c[0][r], c[1][r], c[2][r], c[3][r]);
        ImGui::Separator();
        ImGui::TextWrapped("Left: orbit | Right/Middle: pan | Wheel: zoom\nWASD/QE: move | Esc: quit");
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
