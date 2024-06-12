
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ctsegmentator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_LIST         
    "${ctsegmentator_SOURCE_DIR}/include/ctsegmentator/ctsegmentator.hpp"    
)

add_library(libctsegmentator INTERFACE
    ${HEADER_LIST}
	)

target_include_directories(
    libctsegmentator
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${ctsegmentator_SOURCE_DIR}/include>
)

find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(libctsegmentator INTERFACE "${TORCH_LIBRARIES}")

# Function to copy generated models to target binary directory
# Dependency on models are handled by linking target to libdxmc
function(copy_ctsegmentator_model_to_binary_dir func_target)
        set(MODELS freezed_model1.pt freezed_model2.pt freezed_model3.pt freezed_model4.pt)
        foreach(model_name ${MODELS})                       
            add_custom_command(
                TARGET ${func_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ctsegmentator_SOURCE_DIR}/data/models/${model_name}"
                    "$<TARGET_FILE_DIR:${func_target}>/${model_name}"            
            )
        endforeach()               
endfunction()

function(copy_torch_binaries func_target)
        # The following code block is suggested to be used on Windows.
        # According to https://github.com/pytorch/pytorch/issues/25457,
        # the DLLs need to be copied to avoid memory errors.
        if (MSVC)
                file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
                add_custom_command(TARGET ${func_target}
                                POST_BUILD
                                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                ${TORCH_DLLS}
                                $<TARGET_FILE_DIR:${func_target}>)
        endif (MSVC)
endfunction()

add_subdirectory(tests)


# Download models 
SET(CTSEGMENTATOR_MODELS_DOWNLOAD OFF CACHE BOOL "Download trained models from GitHub release for CT Segmentation.")
if(${CTSEGMENTATOR_MODELS_DOWNLOAD})
    SET(CTSEGMENTATOR_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models" CACHE PATH "Path to directory holding Torch models")
    # Creating directory if it do not exists
    if(NOT EXISTS ${CTSEGMENTATOR_MODELS_DIR})
        file(MAKE_DIRECTORY ${directory})
    endif()

    if(NOT EXISTS "${CTSEGMENTATOR_MODELS_DIR}/freezed_model1.pt")
        MESSAGE("Downloading freezed_model1.pt")
        FILE(DOWNLOAD "https://github.com/medicalphysics/ctsegmentation/releases/download/v0.1/freezed_model1.pt" "${CTSEGMENTATOR_MODELS_DIR}" SHOW_PROGRESS)
    endif()
    if(NOT EXISTS "${CTSEGMENTATOR_MODELS_DIR}/freezed_model2.pt")
        MESSAGE("Downloading freezed_model2.pt")
        FILE(DOWNLOAD "https://github.com/medicalphysics/ctsegmentation/releases/download/v0.1/freezed_model2.pt" "${CTSEGMENTATOR_MODELS_DIR}" SHOW_PROGRESS)
    endif()
    if(NOT EXISTS "${CTSEGMENTATOR_MODELS_DIR}/freezed_model3.pt")
        MESSAGE("Downloading freezed_model3.pt")
        FILE(DOWNLOAD "https://github.com/medicalphysics/ctsegmentation/releases/download/v0.1/freezed_model3.pt" "${CTSEGMENTATOR_MODELS_DIR}" SHOW_PROGRESS)
    endif()
    if(NOT EXISTS "${CTSEGMENTATOR_MODELS_DIR}/freezed_model4.pt")
        MESSAGE("Downloading freezed_model4.pt")
        FILE(DOWNLOAD "https://github.com/medicalphysics/ctsegmentation/releases/download/v0.1/freezed_model4.pt" "${CTSEGMENTATOR_MODELS_DIR}" SHOW_PROGRESS)
    endif()
endif()

