#pragma once
#include <glm/glm.hpp>
#include <cmath>
#include <vector>

// PLACEHOLDER renderer (CPU): traces a sphere and a checkered floor using the same
// ray generation as NeRF. It is shown when no NeRF server is available. The NeRF output
// (volume rendering) has the same signature:
//     (c2w, W, H, focal) -> RGB8 image
inline void renderPlaceholder(const glm::mat4 &c2w, int W, int H, float focal, std::vector<unsigned char> &rgb)
{
    rgb.resize(static_cast<size_t>(W) * H * 3);
    const glm::mat3 R(c2w);
    const glm::vec3 origin(c2w[3]);
    const glm::vec3 light = glm::normalize(glm::vec3(0.5f, 1.0f, 0.4f));

#pragma omp parallel for schedule(static)
    for (int j = 0; j < H; ++j)
    {
        for (int i = 0; i < W; ++i)
        {
            // Ray per pixel (NeRF convention: camera looks along -Z)
            const glm::vec3 d = glm::normalize(R * glm::vec3((i + 0.5f - 0.5f * W) / focal,
                                                              -(j + 0.5f - 0.5f * H) / focal, -1.0f));
            // Sky
            const float t = 0.5f * (d.y + 1.0f);
            glm::vec3 col = (1.0f - t) * glm::vec3(0.9f, 0.9f, 0.95f) + t * glm::vec3(0.35f, 0.55f, 0.9f);
            float tHit = 1e9f;

            // Unit sphere at the origin
            const float b = glm::dot(origin, d);
            const float c = glm::dot(origin, origin) - 1.0f;
            const float disc = b * b - c;
            if (disc > 0.0f)
            {
                const float ts = -b - std::sqrt(disc);
                if (ts > 0.0f)
                {
                    const glm::vec3 n = glm::normalize(origin + ts * d);
                    col = glm::vec3(0.85f, 0.3f, 0.25f) * (0.2f + 0.8f * std::max(glm::dot(n, light), 0.0f));
                    tHit = ts;
                }
            }
            // Floor y = -1
            if (d.y < -1e-4f)
            {
                const float tp = (-1.0f - origin.y) / d.y;
                if (tp > 0.0f && tp < tHit)
                {
                    const glm::vec3 p = origin + tp * d;
                    const bool check = (static_cast<int>(std::floor(p.x)) + static_cast<int>(std::floor(p.z))) & 1;
                    const float fade = std::exp(-0.03f * tp);
                    col = (check ? glm::vec3(0.8f) : glm::vec3(0.25f)) * fade + (1.0f - fade) * glm::vec3(0.8f, 0.85f, 0.95f);
                }
            }
            col = glm::clamp(col, 0.0f, 1.0f);
            unsigned char *px = &rgb[(static_cast<size_t>(j) * W + i) * 3];
            px[0] = static_cast<unsigned char>(col.r * 255.0f);
            px[1] = static_cast<unsigned char>(col.g * 255.0f);
            px[2] = static_cast<unsigned char>(col.b * 255.0f);
        }
    }
}
