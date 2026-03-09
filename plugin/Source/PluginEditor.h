#pragma once

#include "PluginProcessor.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

/**
 * MLX AudioGen plugin editor — full DAW-integrated UI.
 *
 * Phase 4a: Waveform display, loop controls, transport, polished layout
 * Phase 4b: BPM sync, key signature, MIDI trigger, bar-based duration
 */
class MLXAudioGenEditor : public juce::AudioProcessorEditor,
                           private juce::Timer
{
public:
    explicit MLXAudioGenEditor (MLXAudioGenProcessor&);
    ~MLXAudioGenEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void updateUIState();
    void onGenerateClicked();
    void drawWaveform (juce::Graphics& g, juce::Rectangle<int> bounds);

    MLXAudioGenProcessor& proc;

    // --- Instance name ---
    juce::TextEditor instanceNameInput;

    // --- Model & Prompt ---
    juce::ComboBox modelSelector;
    juce::TextEditor promptInput;

    // --- Duration controls ---
    juce::ToggleButton barsModeToggle { "Bars" };
    juce::Slider durationSlider;
    juce::Label durationLabel { {}, "Duration" };
    juce::Slider barsSlider;
    juce::Label barsLabel { {}, "Bars" };

    // --- BPM controls (Phase 4b) ---
    juce::ToggleButton dawBpmToggle { "Sync DAW" };
    juce::Slider bpmSlider;
    juce::Label bpmLabel { {}, "BPM" };
    juce::Label bpmDisplay;

    // --- Key signature (Phase 4b) ---
    juce::ComboBox keySelector;

    // --- MusicGen params ---
    juce::Slider temperatureSlider;
    juce::Slider topKSlider;
    juce::Slider guidanceSlider;

    // --- Stable Audio params ---
    juce::Slider stepsSlider;
    juce::Slider cfgScaleSlider;
    juce::ComboBox samplerSelector;

    // --- Seed ---
    juce::Slider seedSlider;
    juce::ToggleButton randomSeedToggle { "Random" };

    // --- Transport & generation ---
    juce::TextButton generateButton { "Generate" };
    juce::TextButton playButton { "Play" };
    juce::TextButton stopButton { "Stop" };
    juce::ToggleButton loopToggle { "Loop" };
    juce::ToggleButton midiTriggerToggle { "MIDI Trigger" };
    juce::Label statusLabel;
    juce::Label errorLabel;

    // --- Effects (Phase 4d) ---
    juce::ToggleButton fxToggle { "FX" };
    juce::Slider compThresholdSlider;
    juce::Slider compRatioSlider;
    juce::Slider delayTimeSlider;
    juce::Slider delayMixSlider;
    juce::Slider reverbSizeSlider;
    juce::Slider reverbMixSlider;

    // --- Waveform ---
    juce::Rectangle<int> waveformBounds;
    float displayProgress { 0.0f };

    // Colours
    static constexpr juce::uint32 bgColour       = 0xFF0A0A0A;
    static constexpr juce::uint32 panelColour     = 0xFF111111;
    static constexpr juce::uint32 surfaceColour   = 0xFF1A1A1A;
    static constexpr juce::uint32 borderColour    = 0xFF2A2A2A;
    static constexpr juce::uint32 textColour      = 0xFFE8E8E8;
    static constexpr juce::uint32 dimTextColour   = 0xFF888888;
    static constexpr juce::uint32 accentColour    = 0xFFFF6B35;
    static constexpr juce::uint32 accentDimColour = 0x40FF6B35;
    static constexpr juce::uint32 successColour   = 0xFF4ADE80;
    static constexpr juce::uint32 errorColourVal  = 0xFFF87171;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenEditor)
};
