GPU=
ifdef gpus
    GPU=--gpus=$(gpus)
endif
export GPU

doc: FORCE
	cd doc && ./build_docs.sh

docker:
	docker build -t sionna -f DOCKERFILE .

install: FORCE
	pip install .

lint:
	pylint sionna/

run-docker:
	docker run -p 8888:8888 --privileged=true --gpus=all --env NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility -v .:/app -w /app  --rm -it --name SionnaSimRT sionna

test: FORCE
	cd test && pytest

FORCE:
