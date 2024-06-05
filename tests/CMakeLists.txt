include(CTest)
ENABLE_TESTING()

set(TEST_SOURCES 
    "test.cpp"    
)


foreach(sourcefile ${TEST_SOURCES})
    get_filename_component(filename ${sourcefile} NAME_WLE)
    add_executable(${filename} ${sourcefile})    
    target_link_libraries(${filename} dxmcctseg )
    add_test(NAME ${filename} COMMAND ${filename})    
    
    #Adding models    
    copy_model_to_binary_dir(${filename} freezed_model1.pt)   
    copy_model_to_binary_dir(${filename} freezed_model2.pt)   
    copy_model_to_binary_dir(${filename} freezed_model3.pt)   
    copy_model_to_binary_dir(${filename} freezed_model4.pt)   

    #Adding torch binaries
    copy_torch_binaries(${filename})
    

endforeach()
