#pragma once

#include "PluginProcessor.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

class MLXAudioGenEditor : public juce::AudioProcessorEditor,
                           private juce::Timer,
                           public juce::DragAndDropContainer
{
public:
    explicit MLXAudioGenEditor (MLXAudioGenProcessor&);
    ~MLXAudioGenEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseWheelMove (const juce::MouseEvent&, const juce::MouseWheelDetails&) override;

private:
    void timerCallback() override;
    void updateUIState();
    void onGenerateClicked();
    void drawWaveform (juce::Graphics& g, juce::Rectangle<int> bounds);

    MLXAudioGenProcessor& proc;

    // Instance name
    juce::TextEditor instanceNameInput;

    // Model & Prompt
    juce::ComboBox modelSelector, keySelector;
    juce::TextEditor promptInput;

    // Prompt templates (5.6)
    juce::ComboBox templateSelector;
    juce::TextButton saveTemplateBtn { "+" };

    // Duration
    juce::ToggleButton barsModeToggle { "Bars" };
    juce::Slider durationSlider, barsSlider;
    juce::Label durationLabel { {}, "Dur" }, barsLabel { {}, "Bars" };

    // BPM
    juce::ToggleButton dawBpmToggle { "Sync DAW" };
    juce::Slider bpmSlider;
    juce::Label bpmLabel { {}, "BPM" }, bpmDisplay;

    // MusicGen / Stable Audio params
    juce::Slider temperatureSlider, topKSlider, guidanceSlider;
    juce::Slider stepsSlider, cfgScaleSlider;
    juce::ComboBox samplerSelector;
    juce::Slider seedSlider;

    // Transport + generation
    juce::TextButton generateButton { "Generate" };
    juce::TextButton variationsButton { "4 Variations" };
    juce::TextButton playButton { "Play" }, stopButton { "Stop" };
    juce::TextButton keepButton { "Keep" }, discardButton { "Discard" };
    juce::TextButton dragButton { "Drag" };
    juce::ToggleButton loopToggle { "Loop" }, midiTriggerToggle { "MIDI" };

    // Variation A/B/C/D (5.2 + 5.8)
    juce::TextButton varButtons[MLXAudioGenProcessor::MAX_VARIATIONS];

    // Sidechain (5.7)
    juce::ToggleButton sidechainMelodyToggle { "SC→Melody" };
    juce::ToggleButton sidechainStyleToggle { "SC→Style" };

    // Session sync (5.10)
    juce::TextButton publishSessionBtn { "Publish" };

    // Output gain + crossfade
    juce::Slider outputGainSlider, loopFadeSlider;

    // Effects
    juce::ToggleButton fxToggle { "FX" };
    juce::Slider compThresholdSlider, compRatioSlider;
    juce::Slider delayTimeSlider, delayMixSlider;
    juce::Slider reverbSizeSlider, reverbMixSlider;

    // Trim
    juce::Slider trimStartSlider, trimEndSlider;
    juce::TextButton trimButton { "Trim" };
    juce::Label trimInfoLabel;

    // Preset / Export
    juce::TextButton savePresetButton { "Save" }, loadPresetButton { "Load" };
    juce::TextButton exportAudioButton { "Export" }, setFolderButton { "Folder" };
    juce::Label folderLabel;

    juce::Label statusLabel, errorLabel;
    juce::Rectangle<int> waveformBounds;
    float displayProgress { 0.0f };

    // Waveform zoom/scroll (5.9)
    float waveformZoom { 1.0f };   // 1.0 = full view, 4.0 = 4x zoom
    float waveformScroll { 0.0f }; // 0.0 to 1.0 position

    // APVTS attachments
    using SA = juce::AudioProcessorValueTreeState::SliderAttachment;
    using BA = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using CA = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    std::unique_ptr<CA> modelAttach;
    std::unique_ptr<SA> durationAttach, barsAttach, bpmAttach;
    std::unique_ptr<BA> barsModeAttach, dawBpmAttach;
    std::unique_ptr<SA> tempAttach, topKAttach, guidanceAttach;
    std::unique_ptr<SA> stepsAttach, cfgAttach;
    std::unique_ptr<CA> samplerAttach;
    std::unique_ptr<SA> seedAttach;
    std::unique_ptr<BA> loopAttach, midiAttach, fxAttach;
    std::unique_ptr<SA> compTAttach, compRAttach;
    std::unique_ptr<SA> delayTAttach, delayMxAttach;
    std::unique_ptr<SA> revSizeAttach, revMixAttach;
    std::unique_ptr<SA> gainAttach, fadeAttach;

    static constexpr juce::uint32 bgColour       = 0xFF0A0A0A;
    static constexpr juce::uint32 panelColour     = 0xFF111111;
    static constexpr juce::uint32 surfaceColour   = 0xFF1A1A1A;
    static constexpr juce::uint32 borderColour    = 0xFF2A2A2A;
    static constexpr juce::uint32 textColour      = 0xFFE8E8E8;
    static constexpr juce::uint32 dimTextColour   = 0xFF888888;
    static constexpr juce::uint32 accentColour    = 0xFFFF6B35;
    static constexpr juce::uint32 successColour   = 0xFF4ADE80;
    static constexpr juce::uint32 errorColourVal  = 0xFFF87171;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenEditor)
};
