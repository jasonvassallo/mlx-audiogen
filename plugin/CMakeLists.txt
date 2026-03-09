cmake_minimum_required(VERSION 3.22)

project(MLXAudioGen VERSION 0.1.0)

# Add JUCE
add_subdirectory(JUCE)

# Plugin target
juce_add_plugin(MLXAudioGen
    COMPANY_NAME "Jason Vassallo"
    PLUGIN_MANUFACTURER_CODE Jvas
    PLUGIN_CODE Mxag
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "MLX AudioGen"
    BUNDLE_ID "com.jasonvassallo.mlxaudiogen"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
)

target_sources(MLXAudioGen
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/HttpClient.cpp
        Source/ServerLauncher.cpp
)

target_compile_definitions(MLXAudioGen
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(MLXAudioGen
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
