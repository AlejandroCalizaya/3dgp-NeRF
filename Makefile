# Usage:  make run     (builds the image if needed, compiles and opens the window)
APP_NAME := nerf-viewer
EXE := viewer.out
NERF_HOST ?= host.docker.internal
NERF_PORT ?= 5555
SERVER_HOST ?= 0.0.0.0   # the viewer runs in Docker: the server must listen outside localhost

ifeq ($(OS),Windows_NT)
	IS_WINDOWS := 1
else
	IS_WSL := $(shell uname -r 2>/dev/null | grep -i microsoft)
endif

ifneq ($(strip $(IS_WINDOWS)$(IS_WSL)),)
    DISPLAY ?= host.docker.internal:0
else
    DISPLAY ?= :0
endif

SOURCES := src/main.cpp \
    imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
    imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp

.PHONY: run build bash clean server prepare train video viewer-cv

IMAGE_ID := $(shell docker images -q $(APP_NAME))

run:
	$(if $(IMAGE_ID),,@$(MAKE) build)
	docker run --rm -it \
        -e DISPLAY=$(DISPLAY) \
        -e LIBGL_ALWAYS_SOFTWARE=1 \
        -e NERF_HOST=$(NERF_HOST) -e NERF_PORT=$(NERF_PORT) \
        --add-host=host.docker.internal:host-gateway \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
        -v "$(CURDIR)":/workspace \
        -w /workspace \
        $(APP_NAME) \
        bash -c 'g++ -std=c++17 -O2 -fopenmp -pthread $(SOURCES) /opt/glad_out/src/glad.c \
            -I/opt/glad_out/include -Iimgui -Iimgui/backends \
            $$(pkg-config --cflags --libs glfw3) -lGL -ldl -o $(EXE) && ./$(EXE)'

build:
	docker build -t $(APP_NAME) .

bash:
	docker run --rm -it -e DISPLAY=$(DISPLAY) -e LIBGL_ALWAYS_SOFTWARE=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$(CURDIR)":/workspace -w /workspace $(APP_NAME) bash

clean:
	docker rmi -f $(APP_NAME) || true
	rm -f $(EXE)

# ---- NeRF (run on Windows with the .env environment, not in Docker) ----
# One folder per scene: data/<SCENE>/.  Examples:
#   make prepare SCENE=mug VIDEO=src/videos/mug.mp4     (video -> COLMAP -> transforms.json)
#   make train   SCENE=mug ITERS=10000                  (creates data/mug/nerf.pt)
#   make server                                         (serves ALL scenes data/*/nerf.pt)
#   make run                                            (in the window: "Scene" combo)
PY := .env/python.exe
SCENE ?= casco
VIDEO ?= src/videos/$(SCENE).mp4
ITERS ?= 10000

server:
	$(PY) nerf/server.py --host $(SERVER_HOST) --port $(NERF_PORT)
prepare:
	$(PY) nerf/prepare_data.py --video $(VIDEO) --out data/$(SCENE)
train:
	$(PY) nerf/train.py --data data/$(SCENE) --out data/$(SCENE)/nerf.pt --iters $(ITERS)
video:
	$(PY) nerf/viewer_cv.py --ckpt data/$(SCENE)/nerf.pt --out data/$(SCENE)/orbit.mp4 --video
viewer-cv:
	$(PY) nerf/viewer_cv.py --ckpt data/$(SCENE)/nerf.pt
