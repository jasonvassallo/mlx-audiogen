#include "PluginProcessor.h"
#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// APVTS Parameter Layout
// ---------------------------------------------------------------------------

juce::AudioProcessorValueTreeState::ParameterLayout
MLXAudioGenProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID ("model", 1), "Model",
        juce::StringArray { "MusicGen", "Stable Audio" }, 0));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("seconds", 1), "Duration",
        juce::NormalisableRange<float> (0.5f, 60.0f, 0.5f), 5.0f));
    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("bars", 1), "Bars", 1, 32, 4));
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("barsMode", 1), "Bars Mode", false));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("manualBpm", 1), "Manual BPM",
        juce::NormalisableRange<float> (40.0f, 240.0f, 1.0f), 120.0f));
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("dawBpm", 1), "DAW BPM Sync", true));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("temperature", 1), "Temperature",
        juce::NormalisableRange<float> (0.1f, 2.0f, 0.05f), 1.0f));
    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("topK", 1), "Top K", 1, 500, 250));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("guidance", 1), "Guidance",
        juce::NormalisableRange<float> (0.0f, 10.0f, 0.1f), 3.0f));

    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("steps", 1), "Steps", 1, 100, 8));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("cfgScale", 1), "CFG Scale",
        juce::NormalisableRange<float> (0.0f, 15.0f, 0.1f), 6.0f));
    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID ("sampler", 1), "Sampler",
        juce::StringArray { "Euler", "RK4" }, 0));

    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("seed", 1), "Seed", -1, 99999, -1));
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("loop", 1), "Loop", true));
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("midiTrigger", 1), "MIDI Trigger", false));

    // Effects
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("fxEnabled", 1), "FX Enabled", false));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("compThreshold", 1), "Comp Threshold",
        juce::NormalisableRange<float> (-60.0f, 0.0f, 1.0f), 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("compRatio", 1), "Comp Ratio",
        juce::NormalisableRange<float> (1.0f, 20.0f, 0.1f), 1.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayTime", 1), "Delay Time",
        juce::NormalisableRange<float> (0.0f, 1000.0f, 1.0f), 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayMix", 1), "Delay Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayFeedback", 1), "Delay Feedback",
        juce::NormalisableRange<float> (0.0f, 0.95f, 0.01f), 0.3f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbSize", 1), "Reverb Size",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbDamping", 1), "Reverb Damping",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbMix", 1), "Reverb Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("outputGain", 1), "Output Gain",
        juce::NormalisableRange<float> (-24.0f, 12.0f, 0.1f), 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("loopFade", 1), "Loop Crossfade",
        juce::NormalisableRange<float> (0.0f, 50.0f, 1.0f), 10.0f));

    return { params.begin(), params.end() };
}

#define PARAM_FLOAT(id)  apvts.getRawParameterValue(id)->load()
#define PARAM_INT(id)    (int) apvts.getRawParameterValue(id)->load()
#define PARAM_BOOL(id)   (apvts.getRawParameterValue(id)->load() >= 0.5f)
#define PARAM_CHOICE(id) (int) apvts.getRawParameterValue(id)->load()

// ---------------------------------------------------------------------------
// Generation thread
// ---------------------------------------------------------------------------

class GenerationThread : public juce::Thread
{
public:
    GenerationThread (MLXAudioGenProcessor& p, int varCount = 0)
        : juce::Thread ("MLX Generation"), processor (p), variations (varCount) {}
    void run() override {
        if (variations > 0) processor.runVariations (variations);
        else processor.runGeneration();
    }
private:
    MLXAudioGenProcessor& processor;
    int variations;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

MLXAudioGenProcessor::MLXAudioGenProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Sidechain", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "MLXAudioGen", createParameterLayout())
{
}

MLXAudioGenProcessor::~MLXAudioGenProcessor()
{
    stopTimer();
    if (generationThread && generationThread->isThreadRunning())
        generationThread->stopThread (5000);
}

bool MLXAudioGenProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    auto mainOut = layouts.getMainOutputChannelSet();
    if (mainOut != juce::AudioChannelSet::stereo() &&
        mainOut != juce::AudioChannelSet::mono())
        return false;
    // Sidechain is optional
    auto sideIn = layouts.getChannelSet (true, 0);
    if (! sideIn.isDisabled() && sideIn != juce::AudioChannelSet::stereo() &&
        sideIn != juce::AudioChannelSet::mono())
        return false;
    return true;
}

void MLXAudioGenProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = (juce::uint32) samplesPerBlock;
    spec.numChannels = 2;
    delayLine.prepare (spec);
    delayLine.setMaximumDelayInSamples ((int) sampleRate);
    reverb.prepare (spec);

    // Sidechain recording buffer (max 30 seconds)
    sidechainBuffer.setSize (2, (int) (sampleRate * 30.0));
    sidechainWritePos = 0;

    if (! serverLauncher.isServerAlive())
    {
        auto* launcher = &serverLauncher;
        auto* self = this;
        juce::Thread::launch ([launcher, self] {
            launcher->ensureServerRunning();
            juce::ScopedLock lock (self->stateLock);
            self->statusMessage = launcher->getStatus();
        });
    }

    // Start session sync timer
    startTimerHz (1);
}

void MLXAudioGenProcessor::releaseResources() {}

// ---------------------------------------------------------------------------
// processBlock
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& midi)
{
    buffer.clear();

    // Read DAW BPM
    if (auto* playHead = getPlayHead())
        if (auto pos = playHead->getPosition())
            if (auto bpm = pos->getBpm())
                dawBpm = (float) *bpm;

    // MIDI trigger
    if (PARAM_BOOL ("midiTrigger"))
        for (const auto metadata : midi)
            if (metadata.getMessage().isNoteOn() && ! generating.load())
            { juce::MessageManager::callAsync ([this] { triggerGeneration(); }); break; }

    // Record sidechain input (if enabled and bus is active)
    auto sidechainBus = getBus (true, 0);
    if (sidechainBus != nullptr && sidechainBus->isEnabled() &&
        (useSidechainAsMelody || useSidechainAsStyle))
    {
        recordSidechain (buffer, buffer.getNumSamples());
    }

    // Playback
    const auto& audio = getGeneratedAudio();
    if (! playing.load() || ! hasAudio.load() || audio.getNumSamples() == 0)
        return;

    bool looping = PARAM_BOOL ("loop");
    const int nc = juce::jmin (buffer.getNumChannels(), audio.getNumChannels());
    const int ns = buffer.getNumSamples();
    const int total = audio.getNumSamples();
    int pos = playbackPosition.load();

    for (int ch = 0; ch < nc; ++ch)
    {
        int wp = 0, rp = pos;
        while (wp < ns)
        {
            int tc = juce::jmin (ns - wp, total - rp);
            if (tc > 0) buffer.copyFrom (ch, wp, audio, ch, rp, tc);
            wp += tc; rp += tc;
            if (rp >= total) { if (looping) rp = 0; else { playing.store (false); break; } }
        }
    }

    pos += ns;
    if (pos >= total) pos = looping ? pos % total : total;
    playbackPosition.store (pos);

    // Crossfade
    float fadeMs = PARAM_FLOAT ("loopFade");
    if (looping && fadeMs > 0.0f && total > 0)
    {
        int fadeSamp = juce::jmin ((int) (fadeMs * 0.001f * (float) currentSampleRate), total / 4);
        int cp = playbackPosition.load();
        if (cp >= total - fadeSamp && cp < total)
        {
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                auto* d = buffer.getWritePointer (ch);
                for (int i = 0; i < ns; ++i)
                {
                    int dist = total - (cp - ns + i);
                    if (dist >= 0 && dist < fadeSamp)
                        d[i] *= (float) dist / (float) fadeSamp;
                }
            }
        }
    }

    if (PARAM_BOOL ("fxEnabled")) applyEffects (buffer);

    // Output gain
    float gDb = PARAM_FLOAT ("outputGain");
    if (std::abs (gDb) > 0.05f)
        buffer.applyGain (std::pow (10.0f, gDb / 20.0f));
}

// ---------------------------------------------------------------------------
// Sidechain recording (5.7)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::recordSidechain (const juce::AudioBuffer<float>& buffer, int numSamples)
{
    int nc = juce::jmin (buffer.getNumChannels(), sidechainBuffer.getNumChannels());
    int toWrite = juce::jmin (numSamples, sidechainBuffer.getNumSamples() - sidechainWritePos);
    if (toWrite <= 0) return;

    for (int ch = 0; ch < nc; ++ch)
        sidechainBuffer.copyFrom (ch, sidechainWritePos, buffer, ch, 0, toWrite);
    sidechainWritePos += toWrite;
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::updateEffectsParameters()
{
    juce::dsp::Reverb::Parameters rp;
    rp.roomSize = PARAM_FLOAT ("reverbSize");
    rp.damping = PARAM_FLOAT ("reverbDamping");
    rp.wetLevel = PARAM_FLOAT ("reverbMix");
    rp.dryLevel = 1.0f - rp.wetLevel * 0.5f;
    reverb.setParameters (rp);
    delayLine.setDelay (PARAM_FLOAT ("delayTime") * 0.001f * (float) currentSampleRate);
}

void MLXAudioGenProcessor::applyEffects (juce::AudioBuffer<float>& buffer)
{
    updateEffectsParameters();
    const int nc = buffer.getNumChannels(), ns = buffer.getNumSamples();
    float compT = PARAM_FLOAT ("compThreshold"), compR = PARAM_FLOAT ("compRatio");
    float delayMx = PARAM_FLOAT ("delayMix"), delayFb = PARAM_FLOAT ("delayFeedback");
    float delayT = PARAM_FLOAT ("delayTime");

    if (compR > 1.0f)
        for (int ch = 0; ch < nc; ++ch)
        { auto* d = buffer.getWritePointer (ch);
          for (int i = 0; i < ns; ++i)
          { float a = std::abs (d[i]);
            float db = a > 0.0001f ? 20.0f * std::log10 (a) : -80.0f;
            if (db > compT) d[i] *= std::pow (10.0f, ((compT + (db - compT) / compR) - db) / 20.0f);
          }
        }

    if (delayT > 0.0f && delayMx > 0.0f)
        for (int ch = 0; ch < nc; ++ch)
        { auto* d = buffer.getWritePointer (ch);
          for (int i = 0; i < ns; ++i)
          { float del = delayLine.popSample (ch);
            delayLine.pushSample (ch, d[i] + del * delayFb);
            d[i] = d[i] * (1.0f - delayMx) + del * delayMx;
          }
        }

    if (PARAM_FLOAT ("reverbMix") > 0.0f)
    { juce::dsp::AudioBlock<float> block (buffer);
      juce::dsp::ProcessContextReplacing<float> ctx (block);
      reverb.process (ctx); }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::togglePlayback()
{
    if (! hasAudio.load()) return;
    if (playing.load()) { playing.store (false); return; }
    if (playbackPosition.load() >= getGeneratedAudio().getNumSamples())
        playbackPosition.store (0);
    playing.store (true);
}

void MLXAudioGenProcessor::stopPlayback()
{ playing.store (false); playbackPosition.store (0); }

float MLXAudioGenProcessor::getPlaybackProgress() const
{
    const auto& a = getGeneratedAudio();
    if (! hasAudio.load() || a.getNumSamples() == 0) return 0.0f;
    return (float) playbackPosition.load() / (float) a.getNumSamples();
}

const juce::AudioBuffer<float>& MLXAudioGenProcessor::getGeneratedAudio() const
{
    if (variationCount > 0 && activeVariation >= 0 && activeVariation < variationCount)
        return variations[activeVariation];
    return variations[0]; // Default
}

// ---------------------------------------------------------------------------
// Variations (5.2 + 5.8)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::setActiveVariation (int idx)
{
    if (idx >= 0 && idx < variationCount)
    {
        activeVariation = idx;
        playbackPosition.store (0);
    }
}

// ---------------------------------------------------------------------------
// DAW integration
// ---------------------------------------------------------------------------

float MLXAudioGenProcessor::getEffectiveBpm() const
{ return PARAM_BOOL ("dawBpm") ? dawBpm : PARAM_FLOAT ("manualBpm"); }

float MLXAudioGenProcessor::getEffectiveSeconds() const
{
    if (! PARAM_BOOL ("barsMode")) return PARAM_FLOAT ("seconds");
    float bpm = getEffectiveBpm();
    return (float) PARAM_INT ("bars") * 4.0f * (60.0f / (bpm > 0.0f ? bpm : 120.0f));
}

juce::String MLXAudioGenProcessor::buildFullPrompt() const
{
    auto full = prompt;
    if (keySignature.isNotEmpty()) full += " in " + keySignature;
    float bpm = getEffectiveBpm();
    if (bpm > 0.0f && (PARAM_BOOL ("barsMode") || PARAM_BOOL ("dawBpm")))
        full += ", " + juce::String ((int) bpm) + " BPM";
    return full;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::triggerGeneration()
{
    if (generating.load() || prompt.isEmpty()) {
        if (prompt.isEmpty()) { juce::ScopedLock l (stateLock); lastError = "Prompt required"; }
        return;
    }
    if (! serverLauncher.isServerAlive()) {
        juce::ScopedLock l (stateLock); lastError = "Server not running"; return;
    }

    // Write sidechain to file if needed
    if ((useSidechainAsMelody || useSidechainAsStyle) && sidechainWritePos > 0)
    {
        auto tmpDir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("mlx-audiogen");
        tmpDir.createDirectory();
        sidechainFile = tmpDir.getChildFile ("sidechain.wav");
        // Write recorded sidechain audio
        juce::AudioBuffer<float> rec (sidechainBuffer.getNumChannels(), sidechainWritePos);
        for (int ch = 0; ch < rec.getNumChannels(); ++ch)
            rec.copyFrom (ch, 0, sidechainBuffer, ch, 0, sidechainWritePos);

        sidechainFile.deleteFile();
        auto stream = sidechainFile.createOutputStream();
        if (stream) {
            juce::WavAudioFormat wav;
            auto writer = std::unique_ptr<juce::AudioFormatWriter> (
                wav.createWriterFor (stream.release(), currentSampleRate,
                    (unsigned) rec.getNumChannels(), 32, {}, 0));
            if (writer) writer->writeFromAudioSampleBuffer (rec, 0, rec.getNumSamples());
        }
        sidechainWritePos = 0;
    }

    generating.store (true);
    progress.store (0.0f);
    playing.store (false);
    { juce::ScopedLock l (stateLock); statusMessage = "Submitting..."; lastError = {}; }

    generationThread = std::make_unique<GenerationThread> (*this);
    generationThread->startThread();
}

void MLXAudioGenProcessor::triggerVariations (int count)
{
    if (generating.load() || prompt.isEmpty()) return;
    if (! serverLauncher.isServerAlive()) return;

    generating.store (true);
    progress.store (0.0f);
    playing.store (false);
    { juce::ScopedLock l (stateLock); statusMessage = "Generating " + juce::String (count) + " variations..."; lastError = {}; }

    generationThread = std::make_unique<GenerationThread> (*this, count);
    generationThread->startThread();
}

void MLXAudioGenProcessor::runGeneration()
{
    auto* obj = new juce::DynamicObject();
    bool isStable = PARAM_CHOICE ("model") == 1;
    obj->setProperty ("model", isStable ? "stable_audio" : "musicgen");
    obj->setProperty ("prompt", buildFullPrompt());
    obj->setProperty ("seconds", (double) getEffectiveSeconds());

    if (! isStable) {
        obj->setProperty ("temperature", (double) PARAM_FLOAT ("temperature"));
        obj->setProperty ("top_k", PARAM_INT ("topK"));
        obj->setProperty ("guidance_coef", (double) PARAM_FLOAT ("guidance"));
    } else {
        obj->setProperty ("steps", PARAM_INT ("steps"));
        obj->setProperty ("cfg_scale", (double) PARAM_FLOAT ("cfgScale"));
        obj->setProperty ("sampler", PARAM_CHOICE ("sampler") == 1 ? "rk4" : "euler");
        if (negativePrompt.isNotEmpty()) obj->setProperty ("negative_prompt", negativePrompt);
    }

    // Sidechain conditioning
    if (useSidechainAsMelody && sidechainFile.existsAsFile())
        obj->setProperty ("melody_path", sidechainFile.getFullPathName());
    if (useSidechainAsStyle && sidechainFile.existsAsFile())
        obj->setProperty ("style_audio_path", sidechainFile.getFullPathName());

    int s = PARAM_INT ("seed");
    if (s >= 0) obj->setProperty ("seed", s);

    auto jsonBody = juce::JSON::toString (juce::var (obj));
    auto jobId = httpClient.submitGeneration (jsonBody);

    if (jobId.isEmpty()) {
        juce::ScopedLock l (stateLock); lastError = "Connection failed"; statusMessage = "Error";
        generating.store (false); return;
    }

    { juce::ScopedLock l (stateLock); statusMessage = "Generating..."; }

    for (int i = 0; i < 1200; ++i)
    {
        if (juce::Thread::currentThreadShouldExit()) { generating.store (false); return; }
        juce::Thread::sleep (500);

        auto sj = httpClient.fetchStatus (jobId);
        if (sj.isEmpty()) continue;
        auto parsed = juce::JSON::parse (sj);
        if (auto* so = parsed.getDynamicObject())
        {
            auto st = so->getProperty ("status").toString();
            float pv = (float) so->getProperty ("progress");
            progress.store (pv);

            if (st == "done") {
                { juce::ScopedLock l (stateLock); statusMessage = "Downloading..."; }
                auto wav = httpClient.downloadAudio (jobId);
                if (wav.getSize() > 0) {
                    auto is = std::make_unique<juce::MemoryInputStream> (wav, false);
                    juce::WavAudioFormat fmt;
                    auto rd = std::unique_ptr<juce::AudioFormatReader> (fmt.createReaderFor (is.release(), true));
                    if (rd) {
                        variations[0].setSize ((int) rd->numChannels, (int) rd->lengthInSamples);
                        rd->read (&variations[0], 0, (int) rd->lengthInSamples, 0, true, true);
                        variationCount = 1; activeVariation = 0;
                        playbackPosition.store (0);
                        hasAudio.store (true); playing.store (true); pendingDecision.store (true);
                        float dur = (float) rd->lengthInSamples / (float) rd->sampleRate;
                        { juce::ScopedLock l (stateLock);
                          statusMessage = "Preview — Keep or Discard? (" + juce::String (dur, 1) + "s)"; }
                    } else { juce::ScopedLock l (stateLock); lastError = "WAV decode failed"; statusMessage = "Error"; }
                } else { juce::ScopedLock l (stateLock); lastError = "Download failed"; statusMessage = "Error"; }
                progress.store (1.0f); generating.store (false); return;
            }
            else if (st == "error") {
                auto em = so->getProperty ("error").toString();
                juce::ScopedLock l (stateLock); lastError = em.isNotEmpty() ? em : "Failed"; statusMessage = "Error";
                generating.store (false); return;
            }
            { juce::ScopedLock l (stateLock); statusMessage = "Generating " + juce::String ((int) (pv * 100)) + "%"; }
        }
    }
    { juce::ScopedLock l (stateLock); lastError = "Timeout"; statusMessage = "Error"; }
    generating.store (false);
}

void MLXAudioGenProcessor::runVariations (int count)
{
    count = juce::jmin (count, MAX_VARIATIONS);
    bool isStable = PARAM_CHOICE ("model") == 1;
    auto fullPrompt = buildFullPrompt();
    float secs = getEffectiveSeconds();

    // Submit all variations
    juce::StringArray jobIds;
    for (int v = 0; v < count; ++v)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("model", isStable ? "stable_audio" : "musicgen");
        obj->setProperty ("prompt", fullPrompt);
        obj->setProperty ("seconds", (double) secs);
        obj->setProperty ("seed", juce::Random::getSystemRandom().nextInt (99999));

        if (! isStable) {
            obj->setProperty ("temperature", (double) PARAM_FLOAT ("temperature"));
            obj->setProperty ("top_k", PARAM_INT ("topK"));
            obj->setProperty ("guidance_coef", (double) PARAM_FLOAT ("guidance"));
        } else {
            obj->setProperty ("steps", PARAM_INT ("steps"));
            obj->setProperty ("cfg_scale", (double) PARAM_FLOAT ("cfgScale"));
            obj->setProperty ("sampler", PARAM_CHOICE ("sampler") == 1 ? "rk4" : "euler");
        }

        auto jsonBody = juce::JSON::toString (juce::var (obj));
        auto jobId = httpClient.submitGeneration (jsonBody);
        if (jobId.isNotEmpty()) jobIds.add (jobId);
    }

    if (jobIds.isEmpty()) {
        juce::ScopedLock l (stateLock); lastError = "Failed to submit variations";
        generating.store (false); return;
    }

    // Poll all jobs
    int completed = 0;
    for (int i = 0; i < 2400 && completed < jobIds.size(); ++i)
    {
        if (juce::Thread::currentThreadShouldExit()) { generating.store (false); return; }
        juce::Thread::sleep (500);

        for (int v = 0; v < jobIds.size(); ++v)
        {
            if (variations[v].getNumSamples() > 0) continue; // Already downloaded

            auto sj = httpClient.fetchStatus (jobIds[v]);
            if (sj.isEmpty()) continue;
            auto parsed = juce::JSON::parse (sj);
            if (auto* so = parsed.getDynamicObject())
            {
                auto st = so->getProperty ("status").toString();
                if (st == "done")
                {
                    auto wav = httpClient.downloadAudio (jobIds[v]);
                    if (wav.getSize() > 0) {
                        auto is = std::make_unique<juce::MemoryInputStream> (wav, false);
                        juce::WavAudioFormat fmt;
                        auto rd = std::unique_ptr<juce::AudioFormatReader> (fmt.createReaderFor (is.release(), true));
                        if (rd) {
                            variations[v].setSize ((int) rd->numChannels, (int) rd->lengthInSamples);
                            rd->read (&variations[v], 0, (int) rd->lengthInSamples, 0, true, true);
                            completed++;
                        }
                    }
                }
                else if (st == "error") completed++;
            }
        }

        progress.store ((float) completed / (float) jobIds.size());
        { juce::ScopedLock l (stateLock);
          statusMessage = "Variations: " + juce::String (completed) + "/" + juce::String (jobIds.size()); }
    }

    variationCount = 0;
    for (int v = 0; v < jobIds.size(); ++v)
        if (variations[v].getNumSamples() > 0) variationCount = v + 1;

    if (variationCount > 0) {
        activeVariation = 0;
        playbackPosition.store (0);
        hasAudio.store (true); playing.store (true); pendingDecision.store (true);
        { juce::ScopedLock l (stateLock);
          statusMessage = juce::String (variationCount) + " variations ready — audition with A/B/C/D"; }
    }

    progress.store (1.0f); generating.store (false);
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

juce::String MLXAudioGenProcessor::getStatusMessage() const
{ juce::ScopedLock l (stateLock); return statusMessage; }

juce::String MLXAudioGenProcessor::getLastError() const
{ juce::ScopedLock l (stateLock); return lastError; }

void MLXAudioGenProcessor::timerCallback()
{
    readSessionState(); // Sync with other instances (5.10)
}

// ---------------------------------------------------------------------------
// Prompt templates (5.6)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::addPromptTemplate (const juce::String& t)
{ if (t.isNotEmpty() && ! promptTemplates.contains (t)) promptTemplates.add (t); }

void MLXAudioGenProcessor::removePromptTemplate (int idx)
{ if (idx >= 0 && idx < promptTemplates.size()) promptTemplates.remove (idx); }

// ---------------------------------------------------------------------------
// Session sync (5.10)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::publishSessionState()
{
    auto dir = juce::File::getSpecialLocation (juce::File::userHomeDirectory)
                   .getChildFile (".mlx-audiogen");
    dir.createDirectory();
    auto file = dir.getChildFile ("session.json");

    auto* obj = new juce::DynamicObject();
    obj->setProperty ("bpm", (double) getEffectiveBpm());
    obj->setProperty ("keySignature", keySignature);
    obj->setProperty ("instanceName", instanceName);
    obj->setProperty ("timestamp", juce::Time::currentTimeMillis());

    file.replaceWithText (juce::JSON::toString (juce::var (obj)));
}

void MLXAudioGenProcessor::readSessionState()
{
    auto file = juce::File::getSpecialLocation (juce::File::userHomeDirectory)
                    .getChildFile (".mlx-audiogen/session.json");
    if (! file.existsAsFile()) return;

    auto text = file.loadFileAsString();
    auto parsed = juce::JSON::parse (text);
    if (auto* obj = parsed.getDynamicObject())
    {
        // Only sync if another instance wrote it (check instance name)
        auto name = obj->getProperty ("instanceName").toString();
        if (name.isNotEmpty() && name != instanceName)
        {
            auto key = obj->getProperty ("keySignature").toString();
            if (key.isNotEmpty() && keySignature.isEmpty())
                keySignature = key; // Adopt key from other instance if we don't have one
        }
    }
}

// ---------------------------------------------------------------------------
// Beat-grid trimmer
// ---------------------------------------------------------------------------

float MLXAudioGenProcessor::getSixteenthNoteSamples() const
{ float bpm = getEffectiveBpm(); return (60.0f / (bpm > 0 ? bpm : 120.0f) / 4.0f) * (float) currentSampleRate; }

float MLXAudioGenProcessor::getTotalBeats() const
{
    const auto& a = getGeneratedAudio();
    if (! hasAudio.load() || a.getNumSamples() == 0) return 0.0f;
    float bpm = getEffectiveBpm();
    return (float) a.getNumSamples() / (float) currentSampleRate * (bpm > 0 ? bpm : 120.0f) / 60.0f;
}

int MLXAudioGenProcessor::getTrimStartSamples() const
{ return juce::jmax (0, (int) (std::round (trimStartBeats * 4.0f) * getSixteenthNoteSamples())); }

int MLXAudioGenProcessor::getTrimEndSamples() const
{
    const auto& a = getGeneratedAudio();
    if (! hasAudio.load()) return 0;
    if (trimEndBeats < 0.0f) return a.getNumSamples();
    return juce::jmin (a.getNumSamples(), (int) (std::round (trimEndBeats * 4.0f) * getSixteenthNoteSamples()));
}

void MLXAudioGenProcessor::applyTrim()
{
    if (! hasAudio.load()) return;
    const auto& src = getGeneratedAudio();
    int start = getTrimStartSamples(), end = getTrimEndSamples();
    if (start >= end) return;
    int len = end - start, nc = src.getNumChannels();
    juce::AudioBuffer<float> trimmed (nc, len);
    for (int ch = 0; ch < nc; ++ch) trimmed.copyFrom (ch, 0, src, ch, start, len);
    variations[activeVariation] = std::move (trimmed);
    playbackPosition.store (0); trimStartBeats = 0.0f; trimEndBeats = -1.0f;
    { juce::ScopedLock l (stateLock);
      statusMessage = "Trimmed to " + juce::String ((float) len / (float) currentSampleRate, 3) + "s"; }
}

// ---------------------------------------------------------------------------
// Keep / Discard / Export
// ---------------------------------------------------------------------------

juce::File MLXAudioGenProcessor::writeTempAudio()
{
    const auto& a = getGeneratedAudio();
    if (! hasAudio.load() || a.getNumSamples() == 0) return {};
    auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("mlx-audiogen");
    dir.createDirectory();
    auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
    auto file = dir.getChildFile (instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav");
    exportAudio (file);
    lastTempFile = file;
    return file;
}

juce::File MLXAudioGenProcessor::keepAudio()
{
    if (! hasAudio.load()) return {};
    juce::File dest;
    if (exportFolder.isNotEmpty()) {
        auto dir = juce::File (exportFolder);
        if (dir.isDirectory()) {
            auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
            dest = dir.getChildFile (instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav");
        }
    }
    if (dest == juce::File()) dest = writeTempAudio();
    else exportAudio (dest);
    pendingDecision.store (false);
    { juce::ScopedLock l (stateLock); statusMessage = "Saved: " + dest.getFileName(); }
    return dest;
}

void MLXAudioGenProcessor::discardAudio()
{
    playing.store (false); hasAudio.store (false); playbackPosition.store (0);
    for (int v = 0; v < MAX_VARIATIONS; ++v) variations[v].setSize (0, 0);
    variationCount = 0; pendingDecision.store (false);
    if (lastTempFile.existsAsFile()) lastTempFile.deleteFile();
    { juce::ScopedLock l (stateLock); statusMessage = "Discarded"; }
}

void MLXAudioGenProcessor::savePreset (const juce::File& file)
{ juce::MemoryBlock b; getStateInformation (b);
  file.replaceWithText (juce::String::fromUTF8 ((const char*) b.getData(), (int) b.getSize())); }

void MLXAudioGenProcessor::loadPreset (const juce::File& file)
{ auto t = file.loadFileAsString();
  if (t.isNotEmpty()) setStateInformation (t.toRawUTF8(), t.getNumBytesAsUTF8()); }

void MLXAudioGenProcessor::exportAudio (const juce::File& file)
{
    const auto& a = getGeneratedAudio();
    if (! hasAudio.load() || a.getNumSamples() == 0) return;
    file.deleteFile();
    auto stream = file.createOutputStream();
    if (! stream) return;
    juce::WavAudioFormat wav;
    auto writer = std::unique_ptr<juce::AudioFormatWriter> (
        wav.createWriterFor (stream.release(), currentSampleRate,
            (unsigned) a.getNumChannels(), 32, {}, 0));
    if (writer) writer->writeFromAudioSampleBuffer (a, 0, a.getNumSamples());
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    state.setProperty ("instanceName", instanceName, nullptr);
    state.setProperty ("prompt", prompt, nullptr);
    state.setProperty ("negativePrompt", negativePrompt, nullptr);
    state.setProperty ("exportFolder", exportFolder, nullptr);
    state.setProperty ("keySignature", keySignature, nullptr);
    state.setProperty ("serverUrl", httpClient.getBaseUrl(), nullptr);
    state.setProperty ("useSidechainAsMelody", useSidechainAsMelody, nullptr);
    state.setProperty ("useSidechainAsStyle", useSidechainAsStyle, nullptr);

    // Save prompt templates
    juce::String templates;
    for (auto& t : promptTemplates) templates += t + "\n";
    state.setProperty ("promptTemplates", templates, nullptr);

    auto xml = state.createXml();
    if (xml) copyXmlToBinary (*xml, destData);
}

juce::AudioProcessorEditor* MLXAudioGenProcessor::createEditor()
{ return new MLXAudioGenEditor (*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{ return new MLXAudioGenProcessor(); }

void MLXAudioGenProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary (data, sizeInBytes);
    if (! xml) return;
    auto state = juce::ValueTree::fromXml (*xml);
    if (! state.isValid()) return;

    apvts.replaceState (state);
    instanceName = state.getProperty ("instanceName", "MLX AudioGen").toString();
    prompt = state.getProperty ("prompt", "").toString();
    negativePrompt = state.getProperty ("negativePrompt", "").toString();
    exportFolder = state.getProperty ("exportFolder", "").toString();
    keySignature = state.getProperty ("keySignature", "").toString();
    useSidechainAsMelody = (bool) state.getProperty ("useSidechainAsMelody", false);
    useSidechainAsStyle = (bool) state.getProperty ("useSidechainAsStyle", false);

    auto templates = state.getProperty ("promptTemplates", "").toString();
    promptTemplates.clear();
    promptTemplates.addTokens (templates, "\n", "");
    promptTemplates.removeEmptyStrings();

    auto url = state.getProperty ("serverUrl", "").toString();
    if (url.isNotEmpty()) httpClient.setBaseUrl (url);
}
