#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>

// Orbit camera around 'target'. Exposes the camera-to-world pose (c2w),
// which is exactly what NeRF needs to generate rays.
// NeRF/OpenGL convention: the camera looks along -Z, +Y is up, +X is right.
class OrbitCamera
{
public:
    glm::vec3 target{0.0f};
    float distance = 4.0f;
    float yaw = 0.6f;    // radians, around Y
    float pitch = 0.35f; // radians, elevation
    float fovY = glm::radians(50.0f);

    // Initialize the orbit from a c2w pose and the point it looks at
    void setFromPose(const glm::mat4 &c2w, const glm::vec3 &tgt, float fov)
    {
        target = tgt;
        fovY = fov;
        const glm::vec3 v = glm::vec3(c2w[3]) - tgt;
        distance = glm::length(v);
        yaw = std::atan2(v.x, v.z);
        pitch = std::asin(std::clamp(v.y / distance, -1.0f, 1.0f));
    }

    void orbit(float dx, float dy)
    {
        yaw -= dx * 0.005f;
        pitch = std::clamp(pitch + dy * 0.005f, -1.5f, 1.5f);
    }

    void pan(float dx, float dy)
    {
        const glm::mat4 c = c2w();
        const float s = distance * 0.0015f;
        target += (-glm::vec3(c[0]) * dx + glm::vec3(c[1]) * dy) * s;
    }

    void zoom(float scroll) { distance = std::clamp(distance * std::pow(0.9f, scroll), 0.3f, 50.0f); }

    // WASD-style movement in the camera plane (moves the target forward/back)
    void move(float forward, float right, float up)
    {
        const glm::mat4 c = c2w();
        target += -glm::vec3(c[2]) * forward + glm::vec3(c[0]) * right + glm::vec3(0, 1, 0) * up;
    }

    glm::vec3 position() const
    {
        return target + distance * glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                             std::cos(pitch) * std::cos(yaw));
    }

    // Camera-to-world pose (columns: right, up, back, position)
    glm::mat4 c2w() const
    {
        const glm::vec3 pos = position();
        const glm::vec3 back = glm::normalize(pos - target);
        const glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), back));
        const glm::vec3 up = glm::cross(back, right);
        return glm::mat4(glm::vec4(right, 0), glm::vec4(up, 0), glm::vec4(back, 0), glm::vec4(pos, 1));
    }

    // Focal length in pixels for an image of height 'height' (pinhole model)
    float focal(int height) const { return 0.5f * height / std::tan(0.5f * fovY); }
};
