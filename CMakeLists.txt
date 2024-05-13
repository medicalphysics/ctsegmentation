
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ctsegmentation)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 20)



# Function to copy generated physicslist binary file to target binary directory
# Dependency on physicslist.bin file is handles by linking target to libdxmc
function(copy_model_to_binary_dir func_target model_name)
        add_custom_command(
            TARGET ${func_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/data/models/${model_name}"
                "$<TARGET_FILE_DIR:${func_target}>/${model_name}"            
        )        
endfunction()

copy_model_to_binary_dir(test test.pt)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:test>)
endif (MSVC)