#include "PluginEditor.h"

static void ss (juce::Slider& s) {
    s.setSliderStyle (juce::Slider::LinearHorizontal);
    s.setTextBoxStyle (juce::Slider::TextBoxRight, false, 50, 18);
    s.setColour (juce::Slider::backgroundColourId,    juce::Colour (0xFF2A2A2A));
    s.setColour (juce::Slider::thumbColourId,          juce::Colour (0xFFFF6B35));
    s.setColour (juce::Slider::trackColourId,          juce::Colour (0xFFFF6B35).withAlpha (0.5f));
    s.setColour (juce::Slider::textBoxTextColourId,    juce::Colour (0xFFE8E8E8));
    s.setColour (juce::Slider::textBoxOutlineColourId, juce::Colour (0xFF2A2A2A));
}
static void st (juce::ToggleButton& t) {
    t.setColour (juce::ToggleButton::textColourId, juce::Colour (0xFF888888));
    t.setColour (juce::ToggleButton::tickColourId, juce::Colour (0xFFFF6B35));
}
static void sc (juce::ComboBox& c) {
    c.setColour (juce::ComboBox::backgroundColourId, juce::Colour (0xFF1A1A1A));
    c.setColour (juce::ComboBox::textColourId, juce::Colour (0xFFE8E8E8));
    c.setColour (juce::ComboBox::outlineColourId, juce::Colour (0xFF2A2A2A));
}
static void sb (juce::TextButton& b) {
    b.setColour (juce::TextButton::buttonColourId, juce::Colour (0xFF1A1A1A));
    b.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFFE8E8E8));
}

static const juce::StringArray KEYS = {
    "", "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor"
};

MLXAudioGenEditor::MLXAudioGenEditor (MLXAudioGenProcessor& p)
    : AudioProcessorEditor (&p), proc (p)
{
    setSize (540, 980);

    // Instance name
    instanceNameInput.setColour (juce::TextEditor::backgroundColourId, juce::Colour (bgColour));
    instanceNameInput.setColour (juce::TextEditor::textColourId, juce::Colour (textColour));
    instanceNameInput.setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    instanceNameInput.setFont (juce::Font (14.0f, juce::Font::bold));
    instanceNameInput.setText (proc.instanceName);
    instanceNameInput.onTextChange = [this] { proc.instanceName = instanceNameInput.getText(); };
    addAndMakeVisible (instanceNameInput);

    // Model + Key
    modelSelector.addItem ("MusicGen", 1);
    modelSelector.addItem ("Stable Audio", 2);
    sc (modelSelector); addAndMakeVisible (modelSelector);
    modelAttach = std::make_unique<CA> (proc.apvts, "model", modelSelector);
    modelSelector.onChange = [this] { updateUIState(); };

    keySelector.addItem ("(no key)", 1);
    for (int i = 1; i < KEYS.size(); ++i) keySelector.addItem (KEYS[i], i + 1);
    int ki = KEYS.indexOf (proc.keySignature);
    keySelector.setSelectedId (ki >= 0 ? ki + 1 : 1);
    keySelector.onChange = [this] {
        int idx = keySelector.getSelectedId() - 1;
        proc.keySignature = (idx > 0 && idx < KEYS.size()) ? KEYS[idx] : "";
    };
    sc (keySelector); addAndMakeVisible (keySelector);

    // Prompt
    promptInput.setMultiLine (true);
    promptInput.setReturnKeyStartsNewLine (false);
    promptInput.setTextToShowWhenEmpty ("Describe the audio...", juce::Colour (dimTextColour));
    promptInput.setColour (juce::TextEditor::backgroundColourId, juce::Colour (surfaceColour));
    promptInput.setColour (juce::TextEditor::textColourId, juce::Colour (textColour));
    promptInput.setColour (juce::TextEditor::outlineColourId, juce::Colour (borderColour));
    promptInput.setColour (juce::TextEditor::focusedOutlineColourId, juce::Colour (accentColour));
    promptInput.setText (proc.prompt);
    promptInput.onTextChange = [this] { proc.prompt = promptInput.getText(); };
    addAndMakeVisible (promptInput);

    // Prompt templates (5.6)
    sc (templateSelector);
    templateSelector.addItem ("(templates)", 1);
    for (int i = 0; i < proc.promptTemplates.size(); ++i)
        templateSelector.addItem (proc.promptTemplates[i], i + 2);
    templateSelector.setSelectedId (1);
    templateSelector.onChange = [this] {
        int idx = templateSelector.getSelectedId() - 2;
        if (idx >= 0 && idx < proc.promptTemplates.size()) {
            proc.prompt = proc.promptTemplates[idx];
            promptInput.setText (proc.prompt);
        }
    };
    addAndMakeVisible (templateSelector);

    sb (saveTemplateBtn);
    saveTemplateBtn.onClick = [this] {
        if (proc.prompt.isNotEmpty()) {
            proc.addPromptTemplate (proc.prompt);
            templateSelector.addItem (proc.prompt, proc.promptTemplates.size() + 1);
        }
    };
    addAndMakeVisible (saveTemplateBtn);

    // Duration
    st (barsModeToggle); addAndMakeVisible (barsModeToggle);
    barsModeAttach = std::make_unique<BA> (proc.apvts, "barsMode", barsModeToggle);
    barsModeToggle.onClick = [this] { updateUIState(); };

    ss (durationSlider); addAndMakeVisible (durationSlider);
    durationAttach = std::make_unique<SA> (proc.apvts, "seconds", durationSlider);
    durationLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    durationLabel.setFont (juce::Font (11.0f));
    addAndMakeVisible (durationLabel);

    ss (barsSlider); addAndMakeVisible (barsSlider);
    barsAttach = std::make_unique<SA> (proc.apvts, "bars", barsSlider);
    barsLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    barsLabel.setFont (juce::Font (11.0f));
    addAndMakeVisible (barsLabel);

    // BPM
    st (dawBpmToggle); addAndMakeVisible (dawBpmToggle);
    dawBpmAttach = std::make_unique<BA> (proc.apvts, "dawBpm", dawBpmToggle);
    dawBpmToggle.onClick = [this] { updateUIState(); };
    ss (bpmSlider); addAndMakeVisible (bpmSlider);
    bpmAttach = std::make_unique<SA> (proc.apvts, "manualBpm", bpmSlider);
    bpmLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    addAndMakeVisible (bpmLabel);
    bpmDisplay.setColour (juce::Label::textColourId, juce::Colour (accentColour));
    bpmDisplay.setFont (juce::Font (13.0f, juce::Font::bold));
    bpmDisplay.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (bpmDisplay);

    // Model params
    ss (temperatureSlider); addAndMakeVisible (temperatureSlider);
    tempAttach = std::make_unique<SA> (proc.apvts, "temperature", temperatureSlider);
    ss (topKSlider); addAndMakeVisible (topKSlider);
    topKAttach = std::make_unique<SA> (proc.apvts, "topK", topKSlider);
    ss (guidanceSlider); addAndMakeVisible (guidanceSlider);
    guidanceAttach = std::make_unique<SA> (proc.apvts, "guidance", guidanceSlider);
    ss (stepsSlider); addAndMakeVisible (stepsSlider);
    stepsAttach = std::make_unique<SA> (proc.apvts, "steps", stepsSlider);
    ss (cfgScaleSlider); addAndMakeVisible (cfgScaleSlider);
    cfgAttach = std::make_unique<SA> (proc.apvts, "cfgScale", cfgScaleSlider);
    samplerSelector.addItem ("Euler", 1); samplerSelector.addItem ("RK4", 2);
    sc (samplerSelector); addAndMakeVisible (samplerSelector);
    samplerAttach = std::make_unique<CA> (proc.apvts, "sampler", samplerSelector);

    // Seed
    ss (seedSlider); addAndMakeVisible (seedSlider);
    seedAttach = std::make_unique<SA> (proc.apvts, "seed", seedSlider);

    // Transport
    st (loopToggle); addAndMakeVisible (loopToggle);
    loopAttach = std::make_unique<BA> (proc.apvts, "loop", loopToggle);
    st (midiTriggerToggle); addAndMakeVisible (midiTriggerToggle);
    midiAttach = std::make_unique<BA> (proc.apvts, "midiTrigger", midiTriggerToggle);

    generateButton.setColour (juce::TextButton::buttonColourId, juce::Colour (accentColour));
    generateButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    generateButton.onClick = [this] { onGenerateClicked(); };
    addAndMakeVisible (generateButton);

    sb (variationsButton);
    variationsButton.onClick = [this] { proc.triggerVariations (4); };
    addAndMakeVisible (variationsButton);

    sb (playButton); playButton.onClick = [this] { proc.togglePlayback(); };
    addAndMakeVisible (playButton);
    sb (stopButton); stopButton.onClick = [this] { proc.stopPlayback(); };
    addAndMakeVisible (stopButton);

    keepButton.setColour (juce::TextButton::buttonColourId, juce::Colour (successColour));
    keepButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    keepButton.onClick = [this] { proc.keepAudio(); };
    addAndMakeVisible (keepButton);
    discardButton.setColour (juce::TextButton::buttonColourId, juce::Colour (errorColourVal));
    discardButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    discardButton.onClick = [this] { proc.discardAudio(); };
    addAndMakeVisible (discardButton);

    sb (dragButton);
    dragButton.onClick = [this] {
        auto f = proc.writeTempAudio();
        if (f.existsAsFile())
            performExternalDragDropOfFiles (juce::StringArray { f.getFullPathName() }, false, this);
    };
    addAndMakeVisible (dragButton);

    // Variation A/B/C/D buttons (5.2 + 5.8)
    for (int i = 0; i < MLXAudioGenProcessor::MAX_VARIATIONS; ++i)
    {
        varButtons[i].setButtonText (juce::String::charToString ('A' + i));
        sb (varButtons[i]);
        varButtons[i].onClick = [this, i] { proc.setActiveVariation (i); };
        addAndMakeVisible (varButtons[i]);
    }

    // Sidechain (5.7)
    st (sidechainMelodyToggle);
    sidechainMelodyToggle.setToggleState (proc.useSidechainAsMelody, juce::dontSendNotification);
    sidechainMelodyToggle.onClick = [this] { proc.useSidechainAsMelody = sidechainMelodyToggle.getToggleState(); };
    addAndMakeVisible (sidechainMelodyToggle);

    st (sidechainStyleToggle);
    sidechainStyleToggle.setToggleState (proc.useSidechainAsStyle, juce::dontSendNotification);
    sidechainStyleToggle.onClick = [this] { proc.useSidechainAsStyle = sidechainStyleToggle.getToggleState(); };
    addAndMakeVisible (sidechainStyleToggle);

    // Session sync (5.10)
    sb (publishSessionBtn);
    publishSessionBtn.onClick = [this] { proc.publishSessionState(); };
    addAndMakeVisible (publishSessionBtn);

    // Gain + fade
    ss (outputGainSlider); addAndMakeVisible (outputGainSlider);
    gainAttach = std::make_unique<SA> (proc.apvts, "outputGain", outputGainSlider);
    ss (loopFadeSlider); addAndMakeVisible (loopFadeSlider);
    fadeAttach = std::make_unique<SA> (proc.apvts, "loopFade", loopFadeSlider);

    // Effects
    st (fxToggle); addAndMakeVisible (fxToggle);
    fxAttach = std::make_unique<BA> (proc.apvts, "fxEnabled", fxToggle);
    fxToggle.onClick = [this] { updateUIState(); };
    ss (compThresholdSlider); addAndMakeVisible (compThresholdSlider);
    compTAttach = std::make_unique<SA> (proc.apvts, "compThreshold", compThresholdSlider);
    ss (compRatioSlider); addAndMakeVisible (compRatioSlider);
    compRAttach = std::make_unique<SA> (proc.apvts, "compRatio", compRatioSlider);
    ss (delayTimeSlider); addAndMakeVisible (delayTimeSlider);
    delayTAttach = std::make_unique<SA> (proc.apvts, "delayTime", delayTimeSlider);
    ss (delayMixSlider); addAndMakeVisible (delayMixSlider);
    delayMxAttach = std::make_unique<SA> (proc.apvts, "delayMix", delayMixSlider);
    ss (reverbSizeSlider); addAndMakeVisible (reverbSizeSlider);
    revSizeAttach = std::make_unique<SA> (proc.apvts, "reverbSize", reverbSizeSlider);
    ss (reverbMixSlider); addAndMakeVisible (reverbMixSlider);
    revMixAttach = std::make_unique<SA> (proc.apvts, "reverbMix", reverbMixSlider);

    // Trim
    ss (trimStartSlider); trimStartSlider.setRange (0, 32, 0.25);
    trimStartSlider.onValueChange = [this] { proc.trimStartBeats = (float) trimStartSlider.getValue(); };
    addAndMakeVisible (trimStartSlider);
    ss (trimEndSlider); trimEndSlider.setRange (0, 32, 0.25);
    trimEndSlider.onValueChange = [this] { float v = (float) trimEndSlider.getValue(); proc.trimEndBeats = v > 0.0f ? v : -1.0f; };
    addAndMakeVisible (trimEndSlider);
    sb (trimButton); trimButton.onClick = [this] { proc.applyTrim(); };
    addAndMakeVisible (trimButton);
    trimInfoLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    trimInfoLabel.setFont (juce::Font (10.0f));
    addAndMakeVisible (trimInfoLabel);

    // Preset / Export
    sb (savePresetButton);
    savePresetButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Save Preset", juce::File(), "*.mlxpreset");
        c->launchAsync (juce::FileBrowserComponent::saveMode, [this, c] (const auto& fc) {
            auto f = fc.getResult();
            if (f != juce::File()) proc.savePreset (f.withFileExtension ("mlxpreset"));
        });
    };
    addAndMakeVisible (savePresetButton);
    sb (loadPresetButton);
    loadPresetButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Load Preset", juce::File(), "*.mlxpreset");
        c->launchAsync (juce::FileBrowserComponent::openMode, [this, c] (const auto& fc) {
            auto f = fc.getResult();
            if (f != juce::File()) { proc.loadPreset (f); promptInput.setText (proc.prompt); updateUIState(); }
        });
    };
    addAndMakeVisible (loadPresetButton);
    sb (exportAudioButton);
    exportAudioButton.onClick = [this] {
        if (proc.exportFolder.isNotEmpty()) { auto d = juce::File (proc.exportFolder);
            if (d.isDirectory()) { auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
                proc.exportAudio (d.getChildFile (proc.instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav")); return; } }
        auto c = std::make_shared<juce::FileChooser> ("Export", juce::File(), "*.wav");
        c->launchAsync (juce::FileBrowserComponent::saveMode, [this, c] (const auto& fc) {
            auto f = fc.getResult(); if (f != juce::File()) proc.exportAudio (f.withFileExtension ("wav")); });
    };
    addAndMakeVisible (exportAudioButton);
    sb (setFolderButton);
    setFolderButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Export Folder",
            proc.exportFolder.isNotEmpty() ? juce::File (proc.exportFolder) : juce::File());
        c->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
            [this, c] (const auto& fc) { auto f = fc.getResult();
                if (f != juce::File() && f.isDirectory()) {
                    proc.exportFolder = f.getFullPathName();
                    folderLabel.setText (f.getFileName(), juce::dontSendNotification); } }); };
    addAndMakeVisible (setFolderButton);
    folderLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    folderLabel.setFont (juce::Font (10.0f));
    folderLabel.setText (proc.exportFolder.isNotEmpty()
        ? juce::File (proc.exportFolder).getFileName() : "(ask)", juce::dontSendNotification);
    addAndMakeVisible (folderLabel);

    statusLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    statusLabel.setFont (juce::Font (11.0f));
    statusLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (statusLabel);
    errorLabel.setColour (juce::Label::textColourId, juce::Colour (errorColourVal));
    errorLabel.setFont (juce::Font (11.0f));
    errorLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (errorLabel);

    updateUIState();
    startTimerHz (15);
}

MLXAudioGenEditor::~MLXAudioGenEditor() { stopTimer(); }

// ---------------------------------------------------------------------------
// Waveform zoom (5.9) — mouse wheel on waveform area
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& w)
{
    if (waveformBounds.contains (e.getPosition()))
    {
        if (e.mods.isCommandDown())
        {
            // Cmd+scroll = zoom
            waveformZoom = juce::jlimit (1.0f, 16.0f, waveformZoom + w.deltaY * 2.0f);
        }
        else
        {
            // Scroll = pan (only when zoomed in)
            if (waveformZoom > 1.0f)
                waveformScroll = juce::jlimit (0.0f, 1.0f - 1.0f / waveformZoom,
                    waveformScroll - w.deltaY * 0.1f);
        }
        repaint();
    }
    else
    {
        juce::AudioProcessorEditor::mouseWheelMove (e, w);
    }
}

// ---------------------------------------------------------------------------
// Paint
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (bgColour));
    g.setColour (juce::Colour (borderColour));
    g.drawHorizontalLine (waveformBounds.getY() - 4, 14.0f, (float) getWidth() - 14.0f);
    drawWaveform (g, waveformBounds);
    if (proc.isGenerating()) {
        auto bar = getLocalBounds().removeFromBottom (3);
        g.setColour (juce::Colour (borderColour)); g.fillRect (bar);
        g.setColour (juce::Colour (accentColour));
        g.fillRect (bar.removeFromLeft ((int) (bar.getWidth() * displayProgress)));
    }
}

void MLXAudioGenEditor::drawWaveform (juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour (juce::Colour (panelColour));
    g.fillRoundedRectangle (bounds.toFloat(), 4.0f);

    const auto& audio = proc.getGeneratedAudio();
    if (audio.getNumSamples() == 0) {
        g.setColour (juce::Colour (dimTextColour).withAlpha (0.3f));
        g.drawText ("Cmd+Scroll to zoom | Scroll to pan", bounds, juce::Justification::centred);
        return;
    }

    const float* samples = audio.getReadPointer (0);
    const int numSamples = audio.getNumSamples();
    const float w = (float) bounds.getWidth();
    const float h = (float) bounds.getHeight();
    const float midY = (float) bounds.getCentreY();

    // Zoom/scroll: calculate visible sample range
    float visibleFrac = 1.0f / waveformZoom;
    int startSample = (int) (waveformScroll * numSamples);
    int visibleSamples = (int) (visibleFrac * numSamples);
    visibleSamples = juce::jmin (visibleSamples, numSamples - startSample);

    // Waveform
    g.setColour (juce::Colour (accentColour).withAlpha (0.7f));
    juce::Path path;
    for (int x = 0; x < (int) w; ++x) {
        int s0 = startSample + (int) ((float) x / w * visibleSamples);
        int s1 = startSample + juce::jmin ((int) ((float) (x + 1) / w * visibleSamples), visibleSamples);
        s1 = juce::jmin (s1, numSamples);
        float pk = 0.0f;
        for (int s = s0; s < s1; ++s) pk = juce::jmax (pk, std::abs (samples[s]));
        float y = pk * h * 0.45f;
        float px = (float) bounds.getX() + (float) x;
        if (x == 0) path.startNewSubPath (px, midY - y); else path.lineTo (px, midY - y);
    }
    for (int x = (int) w - 1; x >= 0; --x) {
        int s0 = startSample + (int) ((float) x / w * visibleSamples);
        int s1 = startSample + juce::jmin ((int) ((float) (x + 1) / w * visibleSamples), visibleSamples);
        s1 = juce::jmin (s1, numSamples);
        float pk = 0.0f;
        for (int s = s0; s < s1; ++s) pk = juce::jmax (pk, std::abs (samples[s]));
        path.lineTo ((float) bounds.getX() + (float) x, midY + pk * h * 0.45f);
    }
    path.closeSubPath();
    g.fillPath (path);

    // Beat grid
    float totalBeats = proc.getTotalBeats();
    if (totalBeats > 0.0f) {
        float s16total = totalBeats * 4.0f;
        for (int s = 0; s <= (int) s16total; ++s) {
            float samplePos = (float) s / s16total * numSamples;
            if (samplePos < startSample || samplePos > startSample + visibleSamples) continue;
            float frac = (samplePos - startSample) / visibleSamples;
            int lx = bounds.getX() + (int) (frac * w);
            if (s % 16 == 0) g.setColour (juce::Colour (textColour).withAlpha (0.4f));
            else if (s % 4 == 0) g.setColour (juce::Colour (dimTextColour).withAlpha (0.25f));
            else g.setColour (juce::Colour (dimTextColour).withAlpha (0.1f));
            g.drawVerticalLine (lx, (float) bounds.getY(), (float) bounds.getBottom());
        }
    }

    // Trim region
    int ts = proc.getTrimStartSamples(), te = proc.getTrimEndSamples();
    if (ts > 0 || (te < numSamples && te > 0)) {
        float sf = juce::jlimit (0.0f, 1.0f, ((float) ts - startSample) / visibleSamples);
        float ef = juce::jlimit (0.0f, 1.0f, ((float) te - startSample) / visibleSamples);
        g.setColour (juce::Colour (bgColour).withAlpha (0.6f));
        if (sf > 0.0f) g.fillRect (bounds.getX(), bounds.getY(), (int) (sf * w), bounds.getHeight());
        if (ef < 1.0f) g.fillRect (bounds.getX() + (int) (ef * w), bounds.getY(), (int) ((1.0f - ef) * w), bounds.getHeight());
        g.setColour (juce::Colour (successColour));
        if (sf > 0.0f && sf < 1.0f) g.drawVerticalLine (bounds.getX() + (int) (sf * w), (float) bounds.getY(), (float) bounds.getBottom());
        if (ef > 0.0f && ef < 1.0f) g.drawVerticalLine (bounds.getX() + (int) (ef * w), (float) bounds.getY(), (float) bounds.getBottom());
    }

    // Playback position
    if (proc.hasAudioLoaded()) {
        float pp = proc.getPlaybackProgress();
        float samplePos = pp * numSamples;
        if (samplePos >= startSample && samplePos <= startSample + visibleSamples) {
            float frac = (samplePos - startSample) / visibleSamples;
            g.setColour (juce::Colour (textColour));
            g.drawVerticalLine (bounds.getX() + (int) (frac * w), (float) bounds.getY(), (float) bounds.getBottom());
        }
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::resized()
{
    auto area = getLocalBounds().reduced (12);
    const int rh = 20, gap = 3, lw = 65;

    instanceNameInput.setBounds (area.removeFromTop (18)); area.removeFromTop (gap);
    auto topRow = area.removeFromTop (24);
    modelSelector.setBounds (topRow.removeFromLeft (topRow.getWidth() / 2 - 2));
    topRow.removeFromLeft (4);
    keySelector.setBounds (topRow);
    area.removeFromTop (gap);

    promptInput.setBounds (area.removeFromTop (44)); area.removeFromTop (2);

    // Templates row
    auto tmpRow = area.removeFromTop (22);
    templateSelector.setBounds (tmpRow.removeFromLeft (tmpRow.getWidth() - 28));
    tmpRow.removeFromLeft (4);
    saveTemplateBtn.setBounds (tmpRow);
    area.removeFromTop (gap);

    // Duration
    auto dr = area.removeFromTop (rh);
    barsModeToggle.setBounds (dr.removeFromLeft (52));
    if (barsModeToggle.getToggleState()) { barsLabel.setBounds (dr.removeFromLeft (30)); barsSlider.setBounds (dr); }
    else { durationLabel.setBounds (dr.removeFromLeft (25)); durationSlider.setBounds (dr); }
    area.removeFromTop (gap);

    // BPM
    auto br = area.removeFromTop (rh);
    dawBpmToggle.setBounds (br.removeFromLeft (80));
    bpmDisplay.setBounds (br.removeFromRight (55));
    if (! dawBpmToggle.getToggleState()) { bpmLabel.setBounds (br.removeFromLeft (28)); bpmSlider.setBounds (br); }
    area.removeFromTop (gap);

    // Sidechain (5.7)
    auto scRow = area.removeFromTop (rh);
    sidechainMelodyToggle.setBounds (scRow.removeFromLeft (100));
    sidechainStyleToggle.setBounds (scRow.removeFromLeft (90));
    publishSessionBtn.setBounds (scRow);
    area.removeFromTop (gap);

    // Model params
    bool mg = modelSelector.getSelectedId() != 2;
    auto sr = [&] (juce::Slider& s) { auto r = area.removeFromTop (rh); r.removeFromLeft (lw); s.setBounds (r); area.removeFromTop (2); };
    if (mg) { sr (temperatureSlider); sr (topKSlider); sr (guidanceSlider); }
    else { sr (stepsSlider); sr (cfgScaleSlider); auto r = area.removeFromTop (rh); r.removeFromLeft (lw); samplerSelector.setBounds (r); area.removeFromTop (2); }

    auto seedR = area.removeFromTop (rh); seedR.removeFromLeft (lw); seedSlider.setBounds (seedR); area.removeFromTop (gap);

    // Options
    auto optR = area.removeFromTop (rh);
    midiTriggerToggle.setBounds (optR.removeFromLeft (65));
    loopToggle.setBounds (optR.removeFromLeft (55));
    area.removeFromTop (gap);

    // Generate buttons
    auto genRow = area.removeFromTop (28);
    generateButton.setBounds (genRow.removeFromLeft (genRow.getWidth() * 2 / 3 - 2));
    genRow.removeFromLeft (4);
    variationsButton.setBounds (genRow);
    area.removeFromTop (gap);

    // Transport + keep/discard + drag
    auto tr2 = area.removeFromTop (24);
    int bw2 = (tr2.getWidth() - 20) / 5;
    playButton.setBounds (tr2.removeFromLeft (bw2)); tr2.removeFromLeft (3);
    stopButton.setBounds (tr2.removeFromLeft (bw2)); tr2.removeFromLeft (3);
    keepButton.setBounds (tr2.removeFromLeft (bw2)); tr2.removeFromLeft (3);
    discardButton.setBounds (tr2.removeFromLeft (bw2)); tr2.removeFromLeft (3);
    dragButton.setBounds (tr2);
    area.removeFromTop (gap);

    // Variation A/B/C/D
    auto varRow = area.removeFromTop (22);
    int varW = (varRow.getWidth() - 9) / 4;
    for (int i = 0; i < MLXAudioGenProcessor::MAX_VARIATIONS; ++i) {
        varButtons[i].setBounds (varRow.removeFromLeft (varW));
        if (i < 3) varRow.removeFromLeft (3);
    }
    area.removeFromTop (gap);

    // Gain + fade
    sr (outputGainSlider); sr (loopFadeSlider);

    // Trim
    auto t1 = area.removeFromTop (rh); t1.removeFromLeft (lw); trimStartSlider.setBounds (t1); area.removeFromTop (2);
    auto t2 = area.removeFromTop (rh); t2.removeFromLeft (lw); trimEndSlider.setBounds (t2); area.removeFromTop (2);
    auto t3 = area.removeFromTop (16);
    trimButton.setBounds (t3.removeFromLeft (45)); t3.removeFromLeft (4);
    trimInfoLabel.setBounds (t3); area.removeFromTop (gap);

    // FX
    fxToggle.setBounds (area.removeFromTop (rh)); area.removeFromTop (2);
    bool fx = fxToggle.getToggleState();
    if (fx) { sr (compThresholdSlider); sr (compRatioSlider); sr (delayTimeSlider); sr (delayMixSlider); sr (reverbSizeSlider); sr (reverbMixSlider); }

    // Preset
    auto pr = area.removeFromTop (20);
    int bw3 = (pr.getWidth() - 6) / 3;
    savePresetButton.setBounds (pr.removeFromLeft (bw3)); pr.removeFromLeft (3);
    loadPresetButton.setBounds (pr.removeFromLeft (bw3)); pr.removeFromLeft (3);
    exportAudioButton.setBounds (pr); area.removeFromTop (2);
    auto fr = area.removeFromTop (16);
    setFolderButton.setBounds (fr.removeFromLeft (50)); fr.removeFromLeft (4);
    folderLabel.setBounds (fr); area.removeFromTop (gap);

    statusLabel.setBounds (area.removeFromTop (13));
    errorLabel.setBounds (area.removeFromTop (13));
    area.removeFromTop (gap);

    waveformBounds = area;
}

// ---------------------------------------------------------------------------

void MLXAudioGenEditor::updateUIState()
{
    bool mg = modelSelector.getSelectedId() != 2;
    temperatureSlider.setVisible (mg); topKSlider.setVisible (mg); guidanceSlider.setVisible (mg);
    stepsSlider.setVisible (! mg); cfgScaleSlider.setVisible (! mg); samplerSelector.setVisible (! mg);
    durationSlider.setVisible (! barsModeToggle.getToggleState());
    durationLabel.setVisible (! barsModeToggle.getToggleState());
    barsSlider.setVisible (barsModeToggle.getToggleState());
    barsLabel.setVisible (barsModeToggle.getToggleState());
    bpmSlider.setVisible (! dawBpmToggle.getToggleState());
    bpmLabel.setVisible (! dawBpmToggle.getToggleState());
    bool fx = fxToggle.getToggleState();
    compThresholdSlider.setVisible (fx); compRatioSlider.setVisible (fx);
    delayTimeSlider.setVisible (fx); delayMixSlider.setVisible (fx);
    reverbSizeSlider.setVisible (fx); reverbMixSlider.setVisible (fx);
    resized();
}

void MLXAudioGenEditor::timerCallback()
{
    displayProgress = proc.getProgress();
    bool gen = proc.isGenerating();
    generateButton.setEnabled (! gen);
    variationsButton.setEnabled (! gen);
    generateButton.setButtonText (gen ? "Generating " + juce::String ((int) (displayProgress * 100)) + "%" : "Generate");

    bool audio = proc.hasAudioLoaded();
    bool pending = proc.isPendingDecision();
    playButton.setEnabled (audio); stopButton.setEnabled (audio);
    exportAudioButton.setEnabled (audio && ! pending);
    keepButton.setVisible (pending); discardButton.setVisible (pending);
    dragButton.setEnabled (audio);
    playButton.setButtonText (proc.isPlaying() ? "Pause" : "Play");

    // Variation buttons
    int vc = proc.getVariationCount();
    int av = proc.getActiveVariation();
    for (int i = 0; i < MLXAudioGenProcessor::MAX_VARIATIONS; ++i) {
        varButtons[i].setEnabled (i < vc);
        varButtons[i].setColour (juce::TextButton::buttonColourId,
            i == av && vc > 0 ? juce::Colour (accentColour) : juce::Colour (surfaceColour));
        varButtons[i].setColour (juce::TextButton::textColourOffId,
            i == av && vc > 0 ? juce::Colour (0xFF0A0A0A) : juce::Colour (textColour));
    }

    statusLabel.setText (proc.getStatusMessage(), juce::dontSendNotification);
    bpmDisplay.setText (juce::String ((int) proc.getEffectiveBpm()) + " BPM", juce::dontSendNotification);

    if (audio) {
        float tb = proc.getTotalBeats();
        trimStartSlider.setRange (0, (double) tb, 0.25);
        trimEndSlider.setRange (0, (double) tb, 0.25);
        if (trimEndSlider.getValue() == 0.0) trimEndSlider.setValue (tb, juce::dontSendNotification);
        float dur = ((float) trimEndSlider.getValue() - (float) trimStartSlider.getValue()) * 60.0f / proc.getEffectiveBpm();
        trimInfoLabel.setText (juce::String ((float) trimStartSlider.getValue(), 2) + "→"
            + juce::String ((float) trimEndSlider.getValue(), 2) + " (" + juce::String (dur, 3) + "s)",
            juce::dontSendNotification);
    }
    trimButton.setEnabled (audio);

    auto err = proc.getLastError();
    errorLabel.setText (err, juce::dontSendNotification);
    errorLabel.setVisible (err.isNotEmpty());
    repaint();
}

void MLXAudioGenEditor::onGenerateClicked() { proc.triggerGeneration(); }
