CMAKE_GENERATOR ?= "Unix Makefiles"
COMPILE_JOBS ?= 32
CMAKE_ARGS = -DCMAKE_DEPENDS_USE_COMPILER=FALSE -DNUM_BUILDING_JOBS=${COMPILE_JOBS} -G ${CMAKE_GENERATOR} -S.

debug:
	cmake ${CMAKE_ARGS} -Bbuild -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --parallel ${COMPILE_JOBS}

release:
	cmake ${CMAKE_ARGS} -Bbuild-release -DCMAKE_BUILD_TYPE=Release
	cmake --build build-release --parallel ${COMPILE_JOBS}

clean:
	rm -rf build/*
	rm -rf build-release/*

