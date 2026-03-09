#include "PluginProcessor.h"
#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// Background generation thread
// ---------------------------------------------------------------------------

class GenerationThread : public juce::Thread
{
public:
    GenerationThread (MLXAudioGenProcessor& p)
        : juce::Thread ("MLX Generation"), processor (p) {}

    void run() override { processor.runGeneration(); }

private:
    MLXAudioGenProcessor& processor;
};

// ---------------------------------------------------------------------------
// Processor lifecycle
// ---------------------------------------------------------------------------

MLXAudioGenProcessor::MLXAudioGenProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
}

MLXAudioGenProcessor::~MLXAudioGenProcessor()
{
    stopTimer();
    if (generationThread && generationThread->isThreadRunning())
        generationThread->stopThread (5000);
}

void MLXAudioGenProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;

    // Prepare DSP effects
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = (juce::uint32) samplesPerBlock;
    spec.numChannels = 2;

    delayLine.prepare (spec);
    delayLine.setMaximumDelayInSamples ((int) sampleRate); // 1 sec max
    reverb.prepare (spec);

    // Auto-launch server in background
    if (! serverLauncher.isServerAlive())
    {
        auto* launcher = &serverLauncher;
        auto* self = this;
        juce::Thread::launch ([launcher, self]
        {
            launcher->ensureServerRunning();
            juce::ScopedLock lock (self->stateLock);
            self->statusMessage = launcher->getStatus();
        });
    }
}

void MLXAudioGenProcessor::releaseResources() {}

// ---------------------------------------------------------------------------
// Audio processing + MIDI trigger
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& midi)
{
    buffer.clear();

    // Read DAW BPM
    if (auto* playHead = getPlayHead())
    {
        if (auto pos = playHead->getPosition())
        {
            if (auto bpm = pos->getBpm())
                dawBpm = (float) *bpm;
        }
    }

    // MIDI trigger: note-on starts generation
    if (midiTrigger)
    {
        for (const auto metadata : midi)
        {
            auto msg = metadata.getMessage();
            if (msg.isNoteOn() && ! generating.load())
            {
                // Trigger on message thread to avoid thread issues
                juce::MessageManager::callAsync ([this] { triggerGeneration(); });
                break;
            }
        }
    }

    // Playback
    if (! playing.load() || ! hasAudio.load() || generatedAudio.getNumSamples() == 0)
        return;

    const int numChannels = juce::jmin (buffer.getNumChannels(),
                                         generatedAudio.getNumChannels());
    const int numSamples = buffer.getNumSamples();
    const int totalSamples = generatedAudio.getNumSamples();
    int pos = playbackPosition.load();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        int writePos = 0;
        int readPos = pos;

        while (writePos < numSamples)
        {
            int remaining = totalSamples - readPos;
            int toCopy = juce::jmin (numSamples - writePos, remaining);

            if (toCopy > 0)
                buffer.copyFrom (ch, writePos, generatedAudio, ch, readPos, toCopy);

            writePos += toCopy;
            readPos += toCopy;

            if (readPos >= totalSamples)
            {
                if (looping)
                    readPos = 0;
                else
                {
                    playing.store (false);
                    break;
                }
            }
        }
    }

    pos += numSamples;
    if (pos >= totalSamples)
        pos = looping ? pos % totalSamples : totalSamples;
    playbackPosition.store (pos);

    // Apply effects chain
    if (fxEnabled)
        applyEffects (buffer);
}

void MLXAudioGenProcessor::updateEffectsParameters()
{
    // Reverb
    juce::dsp::Reverb::Parameters reverbParams;
    reverbParams.roomSize = reverbSize;
    reverbParams.damping = reverbDamping;
    reverbParams.wetLevel = reverbMix;
    reverbParams.dryLevel = 1.0f - reverbMix * 0.5f;
    reverb.setParameters (reverbParams);

    // Delay
    float delaySamples = delayTime * 0.001f * (float) currentSampleRate;
    delayLine.setDelay (delaySamples);
}

void MLXAudioGenProcessor::applyEffects (juce::AudioBuffer<float>& buffer)
{
    updateEffectsParameters();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Simple 3-band EQ using biquad coefficients
    // (Applied directly to buffer for simplicity)

    // Compressor (simple peak compressor)
    if (compRatio > 1.0f)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer (ch);
            for (int i = 0; i < numSamples; ++i)
            {
                float sample = data[i];
                float absLevel = std::abs (sample);
                float levelDb = absLevel > 0.0001f
                    ? 20.0f * std::log10 (absLevel)
                    : -80.0f;

                if (levelDb > compThreshold)
                {
                    float excess = levelDb - compThreshold;
                    float compressed = compThreshold + excess / compRatio;
                    float gain = std::pow (10.0f, (compressed - levelDb) / 20.0f);
                    data[i] *= gain;
                }
            }
        }
    }

    // Delay (ping-pong style)
    if (delayTime > 0.0f && delayMix > 0.0f)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer (ch);
            for (int i = 0; i < numSamples; ++i)
            {
                float delayed = delayLine.popSample (ch);
                delayLine.pushSample (ch, data[i] + delayed * delayFeedback);
                data[i] = data[i] * (1.0f - delayMix) + delayed * delayMix;
            }
        }
    }

    // Reverb
    if (reverbMix > 0.0f)
    {
        juce::dsp::AudioBlock<float> block (buffer);
        juce::dsp::ProcessContextReplacing<float> context (block);
        reverb.process (context);
    }
}

// ---------------------------------------------------------------------------
// Playback control
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::togglePlayback()
{
    if (! hasAudio.load())
        return;

    if (playing.load())
    {
        playing.store (false);
    }
    else
    {
        if (playbackPosition.load() >= generatedAudio.getNumSamples())
            playbackPosition.store (0);
        playing.store (true);
    }
}

void MLXAudioGenProcessor::stopPlayback()
{
    playing.store (false);
    playbackPosition.store (0);
}

float MLXAudioGenProcessor::getPlaybackProgress() const
{
    if (! hasAudio.load() || generatedAudio.getNumSamples() == 0)
        return 0.0f;
    return (float) playbackPosition.load() / (float) generatedAudio.getNumSamples();
}

// ---------------------------------------------------------------------------
// DAW integration
// ---------------------------------------------------------------------------

float MLXAudioGenProcessor::getEffectiveBpm() const
{
    return useDawBpm ? dawBpm : manualBpm;
}

float MLXAudioGenProcessor::getEffectiveSeconds() const
{
    if (! useBarsMode)
        return seconds;

    // bars × beats_per_bar × seconds_per_beat
    float bpm = getEffectiveBpm();
    if (bpm <= 0.0f) bpm = 120.0f;
    return (float) bars * 4.0f * (60.0f / bpm);
}

juce::String MLXAudioGenProcessor::buildFullPrompt() const
{
    auto fullPrompt = prompt;

    // Append key signature if set
    if (keySignature.isNotEmpty())
        fullPrompt += " in " + keySignature;

    // Append BPM hint for rhythmic content
    if (useBarsMode || useDawBpm)
    {
        float bpm = getEffectiveBpm();
        if (bpm > 0.0f)
            fullPrompt += ", " + juce::String ((int) bpm) + " BPM";
    }

    return fullPrompt;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::triggerGeneration()
{
    if (generating.load())
        return;

    if (prompt.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Prompt is required";
        return;
    }

    if (! serverLauncher.isServerAlive())
    {
        juce::ScopedLock lock (stateLock);
        lastError = serverLauncher.getStatus();
        if (lastError.isEmpty())
            lastError = "Server not running. Waiting for auto-start...";
        return;
    }

    generating.store (true);
    progress.store (0.0f);
    playing.store (false);

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Submitting...";
        lastError = {};
    }

    generationThread = std::make_unique<GenerationThread> (*this);
    generationThread->startThread();
}

void MLXAudioGenProcessor::runGeneration()
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("model", modelType);
    obj->setProperty ("prompt", buildFullPrompt());
    obj->setProperty ("seconds", (double) getEffectiveSeconds());

    if (modelType == "musicgen")
    {
        obj->setProperty ("temperature", (double) temperature);
        obj->setProperty ("top_k", topK);
        obj->setProperty ("guidance_coef", (double) guidanceCoef);
    }
    else
    {
        obj->setProperty ("steps", steps);
        obj->setProperty ("cfg_scale", (double) cfgScale);
        obj->setProperty ("sampler", sampler);
        if (negativePrompt.isNotEmpty())
            obj->setProperty ("negative_prompt", negativePrompt);
    }

    if (seed >= 0)
        obj->setProperty ("seed", seed);

    juce::var json (obj);
    auto jsonBody = juce::JSON::toString (json);

    auto jobId = httpClient.submitGeneration (jsonBody);
    if (jobId.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Failed to connect to server";
        statusMessage = "Error";
        generating.store (false);
        return;
    }

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Generating...";
    }

    // Poll until done
    const int maxPolls = 1200;
    for (int i = 0; i < maxPolls; ++i)
    {
        if (juce::Thread::currentThreadShouldExit())
        {
            generating.store (false);
            return;
        }

        juce::Thread::sleep (500);

        auto statusJson = httpClient.fetchStatus (jobId);
        if (statusJson.isEmpty())
            continue;

        auto parsed = juce::JSON::parse (statusJson);
        if (auto* statusObj = parsed.getDynamicObject())
        {
            auto status = statusObj->getProperty ("status").toString();
            auto progressVal = (float) statusObj->getProperty ("progress");
            progress.store (progressVal);

            if (status == "done")
            {
                {
                    juce::ScopedLock lock (stateLock);
                    statusMessage = "Downloading audio...";
                }

                auto wavData = httpClient.downloadAudio (jobId);
                if (wavData.getSize() > 0)
                {
                    auto inputStream = std::make_unique<juce::MemoryInputStream> (
                        wavData, false);

                    juce::WavAudioFormat wavFormat;
                    auto reader = std::unique_ptr<juce::AudioFormatReader> (
                        wavFormat.createReaderFor (inputStream.release(), true));

                    if (reader != nullptr)
                    {
                        generatedAudio.setSize (
                            (int) reader->numChannels,
                            (int) reader->lengthInSamples);
                        reader->read (&generatedAudio, 0,
                                      (int) reader->lengthInSamples, 0, true, true);
                        playbackPosition.store (0);
                        hasAudio.store (true);
                        playing.store (true); // Auto-play

                        float durSecs = (float) reader->lengthInSamples
                                        / (float) reader->sampleRate;
                        {
                            juce::ScopedLock lock (stateLock);
                            statusMessage = juce::String ("Ready — ")
                                            + juce::String (durSecs, 1) + "s loaded";
                        }
                    }
                    else
                    {
                        juce::ScopedLock lock (stateLock);
                        lastError = "Failed to decode WAV";
                        statusMessage = "Error";
                    }
                }
                else
                {
                    juce::ScopedLock lock (stateLock);
                    lastError = "Failed to download audio";
                    statusMessage = "Error";
                }

                progress.store (1.0f);
                generating.store (false);
                return;
            }
            else if (status == "error")
            {
                auto errorMsg = statusObj->getProperty ("error").toString();
                juce::ScopedLock lock (stateLock);
                lastError = errorMsg.isNotEmpty() ? errorMsg : "Generation failed";
                statusMessage = "Error";
                generating.store (false);
                return;
            }

            {
                juce::ScopedLock lock (stateLock);
                statusMessage = juce::String ("Generating... ")
                                + juce::String ((int) (progressVal * 100)) + "%";
            }
        }
    }

    {
        juce::ScopedLock lock (stateLock);
        lastError = "Generation timed out (10 minutes)";
        statusMessage = "Error";
    }
    generating.store (false);
}

// ---------------------------------------------------------------------------
// Status accessors
// ---------------------------------------------------------------------------

juce::String MLXAudioGenProcessor::getStatusMessage() const
{
    juce::ScopedLock lock (stateLock);
    return statusMessage;
}

juce::String MLXAudioGenProcessor::getLastError() const
{
    juce::ScopedLock lock (stateLock);
    return lastError;
}

void MLXAudioGenProcessor::timerCallback() {}

// ---------------------------------------------------------------------------
// State persistence
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("instanceName", instanceName);
    obj->setProperty ("model", modelType);
    obj->setProperty ("prompt", prompt);
    obj->setProperty ("negativePrompt", negativePrompt);
    obj->setProperty ("seconds", (double) seconds);
    obj->setProperty ("temperature", (double) temperature);
    obj->setProperty ("topK", topK);
    obj->setProperty ("guidanceCoef", (double) guidanceCoef);
    obj->setProperty ("steps", steps);
    obj->setProperty ("cfgScale", (double) cfgScale);
    obj->setProperty ("sampler", sampler);
    obj->setProperty ("seed", seed);
    obj->setProperty ("serverUrl", httpClient.getBaseUrl());
    obj->setProperty ("useDawBpm", useDawBpm);
    obj->setProperty ("manualBpm", (double) manualBpm);
    obj->setProperty ("bars", bars);
    obj->setProperty ("useBarsMode", useBarsMode);
    obj->setProperty ("keySignature", keySignature);
    obj->setProperty ("midiTrigger", midiTrigger);
    obj->setProperty ("looping", looping);
    obj->setProperty ("fxEnabled", fxEnabled);
    obj->setProperty ("eqLowGain", (double) eqLowGain);
    obj->setProperty ("eqMidGain", (double) eqMidGain);
    obj->setProperty ("eqMidFreq", (double) eqMidFreq);
    obj->setProperty ("eqHighGain", (double) eqHighGain);
    obj->setProperty ("compThreshold", (double) compThreshold);
    obj->setProperty ("compRatio", (double) compRatio);
    obj->setProperty ("delayTime", (double) delayTime);
    obj->setProperty ("delayFeedback", (double) delayFeedback);
    obj->setProperty ("delayMix", (double) delayMix);
    obj->setProperty ("reverbSize", (double) reverbSize);
    obj->setProperty ("reverbDamping", (double) reverbDamping);
    obj->setProperty ("reverbMix", (double) reverbMix);

    juce::var json (obj);
    auto text = juce::JSON::toString (json);
    destData.replaceAll (text.toRawUTF8(), text.getNumBytesAsUTF8());
}

juce::AudioProcessorEditor* MLXAudioGenProcessor::createEditor()
{
    return new MLXAudioGenEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MLXAudioGenProcessor();
}

void MLXAudioGenProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto text = juce::String::fromUTF8 (static_cast<const char*> (data), sizeInBytes);
    auto parsed = juce::JSON::parse (text);

    if (auto* obj = parsed.getDynamicObject())
    {
        instanceName = obj->getProperty ("instanceName").toString();
        if (instanceName.isEmpty()) instanceName = "MLX AudioGen";
        modelType = obj->getProperty ("model").toString();
        prompt = obj->getProperty ("prompt").toString();
        negativePrompt = obj->getProperty ("negativePrompt").toString();
        seconds = (float) obj->getProperty ("seconds");
        temperature = (float) obj->getProperty ("temperature");
        topK = (int) obj->getProperty ("topK");
        guidanceCoef = (float) obj->getProperty ("guidanceCoef");
        steps = (int) obj->getProperty ("steps");
        cfgScale = (float) obj->getProperty ("cfgScale");
        sampler = obj->getProperty ("sampler").toString();
        seed = (int) obj->getProperty ("seed");
        useDawBpm = (bool) obj->getProperty ("useDawBpm");
        manualBpm = (float) obj->getProperty ("manualBpm");
        bars = (int) obj->getProperty ("bars");
        useBarsMode = (bool) obj->getProperty ("useBarsMode");
        keySignature = obj->getProperty ("keySignature").toString();
        midiTrigger = (bool) obj->getProperty ("midiTrigger");
        looping = (bool) obj->getProperty ("looping");

        fxEnabled = (bool) obj->getProperty ("fxEnabled");
        eqLowGain = (float) obj->getProperty ("eqLowGain");
        eqMidGain = (float) obj->getProperty ("eqMidGain");
        eqMidFreq = (float) obj->getProperty ("eqMidFreq");
        eqHighGain = (float) obj->getProperty ("eqHighGain");
        compThreshold = (float) obj->getProperty ("compThreshold");
        compRatio = (float) obj->getProperty ("compRatio");
        if (compRatio < 1.0f) compRatio = 1.0f;
        delayTime = (float) obj->getProperty ("delayTime");
        delayFeedback = (float) obj->getProperty ("delayFeedback");
        delayMix = (float) obj->getProperty ("delayMix");
        reverbSize = (float) obj->getProperty ("reverbSize");
        reverbDamping = (float) obj->getProperty ("reverbDamping");
        reverbMix = (float) obj->getProperty ("reverbMix");

        auto serverUrl = obj->getProperty ("serverUrl").toString();
        if (serverUrl.isNotEmpty())
            httpClient.setBaseUrl (serverUrl);
    }
}
