#pragma once

#include "HttpClient.h"
#include "ServerLauncher.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

/**
 * MLX AudioGen plugin processor — complete feature set.
 *
 * All generation parameters exposed via APVTS for Push 2 / automation.
 * Supports variation generation (4 seeds), sidechain input for
 * melody/style conditioning, and plugin-to-plugin session sync.
 */
class MLXAudioGenProcessor : public juce::AudioProcessor,
                              private juce::Timer
{
public:
    MLXAudioGenProcessor();
    ~MLXAudioGenProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "MLX AudioGen"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    // --- APVTS ---
    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // --- Generation ---
    void triggerGeneration();
    void triggerVariations (int count = 4);
    bool isGenerating() const { return generating.load(); }
    float getProgress() const { return progress.load(); }
    juce::String getStatusMessage() const;
    juce::String getLastError() const;
    void runGeneration();
    void runVariations (int count);

    // --- Playback ---
    void togglePlayback();
    void stopPlayback();
    bool isPlaying() const { return playing.load(); }
    bool hasAudioLoaded() const { return hasAudio.load(); }
    float getPlaybackProgress() const;
    const juce::AudioBuffer<float>& getGeneratedAudio() const;

    // --- Variations (5.2 + 5.8 A/B/C/D) ---
    static constexpr int MAX_VARIATIONS = 4;
    int getVariationCount() const { return variationCount; }
    int getActiveVariation() const { return activeVariation; }
    void setActiveVariation (int idx);

    // --- Keep / Discard ---
    juce::File keepAudio();
    void discardAudio();
    bool isPendingDecision() const { return pendingDecision.load(); }
    juce::File writeTempAudio();

    // --- Trim ---
    float trimStartBeats { 0.0f };
    float trimEndBeats { -1.0f };
    int getTrimStartSamples() const;
    int getTrimEndSamples() const;
    float getSixteenthNoteSamples() const;
    float getTotalBeats() const;
    void applyTrim();

    // --- DAW integration ---
    float getEffectiveBpm() const;
    float getEffectiveSeconds() const;

    // --- Prompt templates (5.6) ---
    juce::StringArray promptTemplates;
    void addPromptTemplate (const juce::String& t);
    void removePromptTemplate (int idx);

    // --- Sidechain (5.7) ---
    bool useSidechainAsMelody { false };
    bool useSidechainAsStyle { false };
    juce::File getSidechainFile() const { return sidechainFile; }

    // --- Session sync (5.10) ---
    void publishSessionState();
    void readSessionState();

    // --- Preset / Export ---
    void savePreset (const juce::File& file);
    void loadPreset (const juce::File& file);
    void exportAudio (const juce::File& file);

    // --- Non-automatable state ---
    juce::String instanceName { "MLX AudioGen" };
    juce::String prompt;
    juce::String negativePrompt;
    juce::String exportFolder;
    juce::String keySignature;

    HttpClient httpClient;
    ServerLauncher serverLauncher;

private:
    void timerCallback() override;
    juce::String buildFullPrompt() const;
    void updateEffectsParameters();
    void applyEffects (juce::AudioBuffer<float>& buffer);
    void recordSidechain (const juce::AudioBuffer<float>& buffer, int numSamples);

    // Variation buffers
    juce::AudioBuffer<float> variations[MAX_VARIATIONS];
    int variationCount { 0 };
    int activeVariation { 0 };

    // Playback
    std::atomic<int> playbackPosition { 0 };
    std::atomic<bool> hasAudio { false };
    std::atomic<bool> playing { false };
    std::atomic<bool> pendingDecision { false };
    double currentSampleRate { 44100.0 };
    float dawBpm { 120.0f };

    // Generation
    std::atomic<bool> generating { false };
    std::atomic<float> progress { 0.0f };
    juce::String statusMessage;
    juce::String lastError;
    juce::CriticalSection stateLock;
    std::unique_ptr<juce::Thread> generationThread;

    // DSP
    juce::dsp::DelayLine<float> delayLine { 48000 };
    juce::dsp::Reverb reverb;

    // Sidechain recording
    juce::AudioBuffer<float> sidechainBuffer;
    int sidechainWritePos { 0 };
    juce::File sidechainFile;

    juce::File lastTempFile;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenProcessor)
};
