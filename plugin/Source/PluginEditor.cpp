#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// Slider styling helper
// ---------------------------------------------------------------------------

static void styleSlider (juce::Slider& slider, double min, double max,
                          double interval, double defaultVal,
                          juce::Slider::TextEntryBoxPosition textPos =
                              juce::Slider::TextBoxRight)
{
    slider.setRange (min, max, interval);
    slider.setValue (defaultVal);
    slider.setSliderStyle (juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle (textPos, false, 50, 18);
    slider.setColour (juce::Slider::backgroundColourId,   juce::Colour (0xFF2A2A2A));
    slider.setColour (juce::Slider::thumbColourId,         juce::Colour (0xFFFF6B35));
    slider.setColour (juce::Slider::trackColourId,         juce::Colour (0xFFFF6B35).withAlpha (0.5f));
    slider.setColour (juce::Slider::textBoxTextColourId,   juce::Colour (0xFFE8E8E8));
    slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colour (0xFF2A2A2A));
}

static void styleLabel (juce::Label& label)
{
    label.setColour (juce::Label::textColourId, juce::Colour (0xFF888888));
    label.setFont (juce::Font (11.0f));
}

static void styleToggle (juce::ToggleButton& toggle)
{
    toggle.setColour (juce::ToggleButton::textColourId, juce::Colour (0xFF888888));
    toggle.setColour (juce::ToggleButton::tickColourId, juce::Colour (0xFFFF6B35));
}

// Key signature options
static const juce::StringArray KEY_OPTIONS = {
    "", "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor"
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MLXAudioGenEditor::MLXAudioGenEditor (MLXAudioGenProcessor& p)
    : AudioProcessorEditor (&p), proc (p)
{
    setSize (520, 820);

    // Instance name
    instanceNameInput.setColour (juce::TextEditor::backgroundColourId, juce::Colour (bgColour));
    instanceNameInput.setColour (juce::TextEditor::textColourId, juce::Colour (textColour));
    instanceNameInput.setColour (juce::TextEditor::outlineColourId, juce::Colour (0x00000000));
    instanceNameInput.setFont (juce::Font (14.0f, juce::Font::bold));
    instanceNameInput.setText (proc.instanceName);
    instanceNameInput.onTextChange = [this] { proc.instanceName = instanceNameInput.getText(); };
    addAndMakeVisible (instanceNameInput);

    // Model selector
    modelSelector.addItem ("MusicGen", 1);
    modelSelector.addItem ("Stable Audio", 2);
    modelSelector.setSelectedId (proc.modelType == "stable_audio" ? 2 : 1);
    modelSelector.onChange = [this] {
        proc.modelType = modelSelector.getSelectedId() == 2 ? "stable_audio" : "musicgen";
        updateUIState();
    };
    modelSelector.setColour (juce::ComboBox::backgroundColourId, juce::Colour (surfaceColour));
    modelSelector.setColour (juce::ComboBox::textColourId, juce::Colour (textColour));
    modelSelector.setColour (juce::ComboBox::outlineColourId, juce::Colour (borderColour));
    addAndMakeVisible (modelSelector);

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

    // --- Duration / Bars mode ---
    styleToggle (barsModeToggle);
    barsModeToggle.setToggleState (proc.useBarsMode, juce::dontSendNotification);
    barsModeToggle.onClick = [this] {
        proc.useBarsMode = barsModeToggle.getToggleState();
        updateUIState();
    };
    addAndMakeVisible (barsModeToggle);

    styleSlider (durationSlider, 0.5, 60.0, 0.5, proc.seconds);
    durationSlider.onValueChange = [this] { proc.seconds = (float) durationSlider.getValue(); };
    addAndMakeVisible (durationSlider);
    styleLabel (durationLabel);
    addAndMakeVisible (durationLabel);

    styleSlider (barsSlider, 1, 32, 1, proc.bars);
    barsSlider.onValueChange = [this] { proc.bars = (int) barsSlider.getValue(); };
    addAndMakeVisible (barsSlider);
    styleLabel (barsLabel);
    addAndMakeVisible (barsLabel);

    // --- BPM ---
    styleToggle (dawBpmToggle);
    dawBpmToggle.setToggleState (proc.useDawBpm, juce::dontSendNotification);
    dawBpmToggle.onClick = [this] {
        proc.useDawBpm = dawBpmToggle.getToggleState();
        updateUIState();
    };
    addAndMakeVisible (dawBpmToggle);

    styleSlider (bpmSlider, 40, 240, 1, proc.manualBpm);
    bpmSlider.onValueChange = [this] { proc.manualBpm = (float) bpmSlider.getValue(); };
    addAndMakeVisible (bpmSlider);
    styleLabel (bpmLabel);
    addAndMakeVisible (bpmLabel);

    bpmDisplay.setColour (juce::Label::textColourId, juce::Colour (accentColour));
    bpmDisplay.setFont (juce::Font (13.0f, juce::Font::bold));
    bpmDisplay.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (bpmDisplay);

    // --- Key signature ---
    keySelector.addItem ("(no key)", 1);
    for (int i = 1; i < KEY_OPTIONS.size(); ++i)
        keySelector.addItem (KEY_OPTIONS[i], i + 1);

    int keyIdx = KEY_OPTIONS.indexOf (proc.keySignature);
    keySelector.setSelectedId (keyIdx >= 0 ? keyIdx + 1 : 1);
    keySelector.onChange = [this] {
        int idx = keySelector.getSelectedId() - 1;
        proc.keySignature = (idx > 0 && idx < KEY_OPTIONS.size()) ? KEY_OPTIONS[idx] : "";
    };
    keySelector.setColour (juce::ComboBox::backgroundColourId, juce::Colour (surfaceColour));
    keySelector.setColour (juce::ComboBox::textColourId, juce::Colour (textColour));
    keySelector.setColour (juce::ComboBox::outlineColourId, juce::Colour (borderColour));
    addAndMakeVisible (keySelector);

    // --- MusicGen params ---
    styleSlider (temperatureSlider, 0.1, 2.0, 0.05, proc.temperature);
    temperatureSlider.onValueChange = [this] { proc.temperature = (float) temperatureSlider.getValue(); };
    addAndMakeVisible (temperatureSlider);

    styleSlider (topKSlider, 1, 500, 1, proc.topK);
    topKSlider.onValueChange = [this] { proc.topK = (int) topKSlider.getValue(); };
    addAndMakeVisible (topKSlider);

    styleSlider (guidanceSlider, 0, 10, 0.1, proc.guidanceCoef);
    guidanceSlider.onValueChange = [this] { proc.guidanceCoef = (float) guidanceSlider.getValue(); };
    addAndMakeVisible (guidanceSlider);

    // --- Stable Audio params ---
    styleSlider (stepsSlider, 1, 100, 1, proc.steps);
    stepsSlider.onValueChange = [this] { proc.steps = (int) stepsSlider.getValue(); };
    addAndMakeVisible (stepsSlider);

    styleSlider (cfgScaleSlider, 0, 15, 0.1, proc.cfgScale);
    cfgScaleSlider.onValueChange = [this] { proc.cfgScale = (float) cfgScaleSlider.getValue(); };
    addAndMakeVisible (cfgScaleSlider);

    samplerSelector.addItem ("Euler (fast)", 1);
    samplerSelector.addItem ("RK4 (accurate)", 2);
    samplerSelector.setSelectedId (proc.sampler == "rk4" ? 2 : 1);
    samplerSelector.onChange = [this] {
        proc.sampler = samplerSelector.getSelectedId() == 2 ? "rk4" : "euler";
    };
    samplerSelector.setColour (juce::ComboBox::backgroundColourId, juce::Colour (surfaceColour));
    samplerSelector.setColour (juce::ComboBox::textColourId, juce::Colour (textColour));
    samplerSelector.setColour (juce::ComboBox::outlineColourId, juce::Colour (borderColour));
    addAndMakeVisible (samplerSelector);

    // --- Seed ---
    styleSlider (seedSlider, 0, 99999, 1, proc.seed >= 0 ? proc.seed : 42);
    seedSlider.onValueChange = [this] { proc.seed = (int) seedSlider.getValue(); };
    addAndMakeVisible (seedSlider);

    styleToggle (randomSeedToggle);
    randomSeedToggle.setToggleState (proc.seed < 0, juce::dontSendNotification);
    randomSeedToggle.onClick = [this] {
        proc.seed = randomSeedToggle.getToggleState() ? -1 : (int) seedSlider.getValue();
        updateUIState();
    };
    addAndMakeVisible (randomSeedToggle);

    // --- Generate button ---
    generateButton.setColour (juce::TextButton::buttonColourId, juce::Colour (accentColour));
    generateButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    generateButton.onClick = [this] { onGenerateClicked(); };
    addAndMakeVisible (generateButton);

    // --- Transport ---
    playButton.setColour (juce::TextButton::buttonColourId, juce::Colour (surfaceColour));
    playButton.setColour (juce::TextButton::textColourOffId, juce::Colour (textColour));
    playButton.onClick = [this] { proc.togglePlayback(); };
    addAndMakeVisible (playButton);

    stopButton.setColour (juce::TextButton::buttonColourId, juce::Colour (surfaceColour));
    stopButton.setColour (juce::TextButton::textColourOffId, juce::Colour (textColour));
    stopButton.onClick = [this] { proc.stopPlayback(); };
    addAndMakeVisible (stopButton);

    styleToggle (loopToggle);
    loopToggle.setToggleState (proc.looping, juce::dontSendNotification);
    loopToggle.onClick = [this] { proc.looping = loopToggle.getToggleState(); };
    addAndMakeVisible (loopToggle);

    styleToggle (midiTriggerToggle);
    midiTriggerToggle.setToggleState (proc.midiTrigger, juce::dontSendNotification);
    midiTriggerToggle.onClick = [this] { proc.midiTrigger = midiTriggerToggle.getToggleState(); };
    addAndMakeVisible (midiTriggerToggle);

    // --- Effects ---
    styleToggle (fxToggle);
    fxToggle.setToggleState (proc.fxEnabled, juce::dontSendNotification);
    fxToggle.onClick = [this] { proc.fxEnabled = fxToggle.getToggleState(); updateUIState(); };
    addAndMakeVisible (fxToggle);

    styleSlider (compThresholdSlider, -60, 0, 1, proc.compThreshold);
    compThresholdSlider.setTextValueSuffix (" dB");
    compThresholdSlider.onValueChange = [this] { proc.compThreshold = (float) compThresholdSlider.getValue(); };
    addAndMakeVisible (compThresholdSlider);

    styleSlider (compRatioSlider, 1, 20, 0.1, proc.compRatio);
    compRatioSlider.setTextValueSuffix (":1");
    compRatioSlider.onValueChange = [this] { proc.compRatio = (float) compRatioSlider.getValue(); };
    addAndMakeVisible (compRatioSlider);

    styleSlider (delayTimeSlider, 0, 1000, 1, proc.delayTime);
    delayTimeSlider.setTextValueSuffix (" ms");
    delayTimeSlider.onValueChange = [this] { proc.delayTime = (float) delayTimeSlider.getValue(); };
    addAndMakeVisible (delayTimeSlider);

    styleSlider (delayMixSlider, 0, 1, 0.01, proc.delayMix);
    delayMixSlider.onValueChange = [this] { proc.delayMix = (float) delayMixSlider.getValue(); };
    addAndMakeVisible (delayMixSlider);

    styleSlider (reverbSizeSlider, 0, 1, 0.01, proc.reverbSize);
    reverbSizeSlider.onValueChange = [this] { proc.reverbSize = (float) reverbSizeSlider.getValue(); };
    addAndMakeVisible (reverbSizeSlider);

    styleSlider (reverbMixSlider, 0, 1, 0.01, proc.reverbMix);
    reverbMixSlider.onValueChange = [this] { proc.reverbMix = (float) reverbMixSlider.getValue(); };
    addAndMakeVisible (reverbMixSlider);

    // --- Status ---
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

MLXAudioGenEditor::~MLXAudioGenEditor()
{
    stopTimer();
}

// ---------------------------------------------------------------------------
// Paint — waveform + progress bar
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (bgColour));

    // Section dividers
    auto drawDivider = [&] (int y) {
        g.setColour (juce::Colour (borderColour));
        g.drawHorizontalLine (y, 16.0f, (float) getWidth() - 16.0f);
    };

    drawDivider (waveformBounds.getY() - 4);

    // Waveform
    drawWaveform (g, waveformBounds);

    // Generation progress bar (bottom edge)
    if (proc.isGenerating())
    {
        auto bar = getLocalBounds().removeFromBottom (3);
        g.setColour (juce::Colour (borderColour));
        g.fillRect (bar);
        g.setColour (juce::Colour (accentColour));
        g.fillRect (bar.removeFromLeft ((int) (bar.getWidth() * displayProgress)));
    }
}

void MLXAudioGenEditor::drawWaveform (juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour (juce::Colour (panelColour));
    g.fillRoundedRectangle (bounds.toFloat(), 4.0f);

    const auto& audio = proc.getGeneratedAudio();
    if (audio.getNumSamples() == 0)
    {
        g.setColour (juce::Colour (dimTextColour).withAlpha (0.3f));
        g.drawText ("No audio generated yet", bounds, juce::Justification::centred);
        return;
    }

    // Draw waveform
    const float* samples = audio.getReadPointer (0);
    const int numSamples = audio.getNumSamples();
    const float width = (float) bounds.getWidth();
    const float height = (float) bounds.getHeight();
    const float midY = bounds.getCentreY();

    g.setColour (juce::Colour (accentColour).withAlpha (0.7f));

    juce::Path path;
    for (int x = 0; x < (int) width; ++x)
    {
        // Map pixel to sample range
        int startSample = (int) ((float) x / width * numSamples);
        int endSample = (int) ((float) (x + 1) / width * numSamples);
        endSample = juce::jmin (endSample, numSamples);

        // Find peak in this range
        float maxVal = 0.0f;
        for (int s = startSample; s < endSample; ++s)
            maxVal = juce::jmax (maxVal, std::abs (samples[s]));

        float y = maxVal * height * 0.45f;
        float px = (float) (bounds.getX() + x);

        if (x == 0)
        {
            path.startNewSubPath (px, midY - y);
        }
        else
        {
            path.lineTo (px, midY - y);
        }
    }

    // Mirror bottom
    for (int x = (int) width - 1; x >= 0; --x)
    {
        int startSample = (int) ((float) x / width * numSamples);
        int endSample = (int) ((float) (x + 1) / width * numSamples);
        endSample = juce::jmin (endSample, numSamples);

        float maxVal = 0.0f;
        for (int s = startSample; s < endSample; ++s)
            maxVal = juce::jmax (maxVal, std::abs (samples[s]));

        float y = maxVal * height * 0.45f;
        float px = (float) (bounds.getX() + x);
        path.lineTo (px, midY + y);
    }

    path.closeSubPath();
    g.fillPath (path);

    // Playback position indicator
    if (proc.hasAudioLoaded())
    {
        float playProgress = proc.getPlaybackProgress();
        int lineX = bounds.getX() + (int) (playProgress * width);
        g.setColour (juce::Colour (textColour));
        g.drawVerticalLine (lineX, (float) bounds.getY(), (float) bounds.getBottom());
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::resized()
{
    auto area = getLocalBounds().reduced (14);
    const int rowH = 22;
    const int gap = 5;
    const int labelW = 75;

    // Instance name
    instanceNameInput.setBounds (area.removeFromTop (22));
    area.removeFromTop (3);

    // Model selector
    modelSelector.setBounds (area.removeFromTop (26));
    area.removeFromTop (gap);

    // Key signature (next to model)
    auto keyRow = area.removeFromTop (26);
    auto keyLabelArea = keyRow.removeFromLeft (labelW);
    // Draw "Key" label by just setting bounds
    keySelector.setBounds (keyRow);
    area.removeFromTop (gap);

    // Prompt
    promptInput.setBounds (area.removeFromTop (56));
    area.removeFromTop (gap);

    // Duration mode toggle + controls
    auto durModeRow = area.removeFromTop (rowH);
    barsModeToggle.setBounds (durModeRow.removeFromLeft (60));

    if (proc.useBarsMode)
    {
        barsLabel.setBounds (durModeRow.removeFromLeft (labelW - 60));
        barsSlider.setBounds (durModeRow);
    }
    else
    {
        durationLabel.setBounds (durModeRow.removeFromLeft (labelW - 60));
        durationSlider.setBounds (durModeRow);
    }
    area.removeFromTop (gap);

    // BPM row
    auto bpmRow = area.removeFromTop (rowH);
    dawBpmToggle.setBounds (bpmRow.removeFromLeft (85));
    bpmDisplay.setBounds (bpmRow.removeFromRight (60));

    if (! proc.useDawBpm)
    {
        bpmLabel.setBounds (bpmRow.removeFromLeft (30));
        bpmSlider.setBounds (bpmRow);
    }
    area.removeFromTop (gap);

    // Model-specific params
    bool isMusicGen = proc.modelType != "stable_audio";
    auto makeSliderRow = [&] (juce::Slider& slider) {
        auto row = area.removeFromTop (rowH);
        row.removeFromLeft (labelW);
        slider.setBounds (row);
        area.removeFromTop (3);
    };

    if (isMusicGen)
    {
        makeSliderRow (temperatureSlider);
        makeSliderRow (topKSlider);
        makeSliderRow (guidanceSlider);
    }
    else
    {
        makeSliderRow (stepsSlider);
        makeSliderRow (cfgScaleSlider);
        auto samplerRow = area.removeFromTop (rowH);
        samplerRow.removeFromLeft (labelW);
        samplerSelector.setBounds (samplerRow);
        area.removeFromTop (3);
    }

    // Seed
    auto seedRow = area.removeFromTop (rowH);
    randomSeedToggle.setBounds (seedRow.removeFromLeft (80));
    seedSlider.setBounds (seedRow);
    area.removeFromTop (gap);

    // MIDI trigger + loop
    auto optionsRow = area.removeFromTop (rowH);
    midiTriggerToggle.setBounds (optionsRow.removeFromLeft (110));
    loopToggle.setBounds (optionsRow.removeFromLeft (70));
    area.removeFromTop (gap);

    // Generate button
    generateButton.setBounds (area.removeFromTop (32));
    area.removeFromTop (gap);

    // Transport
    auto transportRow = area.removeFromTop (28);
    playButton.setBounds (transportRow.removeFromLeft (80));
    transportRow.removeFromLeft (4);
    stopButton.setBounds (transportRow.removeFromLeft (80));
    area.removeFromTop (gap);

    // Effects
    fxToggle.setBounds (area.removeFromTop (rowH));
    area.removeFromTop (3);
    if (proc.fxEnabled)
    {
        makeSliderRow (compThresholdSlider);
        makeSliderRow (compRatioSlider);
        makeSliderRow (delayTimeSlider);
        makeSliderRow (delayMixSlider);
        makeSliderRow (reverbSizeSlider);
        makeSliderRow (reverbMixSlider);
    }
    area.removeFromTop (gap);

    // Status / Error
    statusLabel.setBounds (area.removeFromTop (16));
    errorLabel.setBounds (area.removeFromTop (16));
    area.removeFromTop (gap);

    // Waveform gets remaining space
    waveformBounds = area;
}

// ---------------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::updateUIState()
{
    bool isMusicGen = proc.modelType != "stable_audio";

    temperatureSlider.setVisible (isMusicGen);
    topKSlider.setVisible (isMusicGen);
    guidanceSlider.setVisible (isMusicGen);

    stepsSlider.setVisible (! isMusicGen);
    cfgScaleSlider.setVisible (! isMusicGen);
    samplerSelector.setVisible (! isMusicGen);

    durationSlider.setVisible (! proc.useBarsMode);
    durationLabel.setVisible (! proc.useBarsMode);
    barsSlider.setVisible (proc.useBarsMode);
    barsLabel.setVisible (proc.useBarsMode);

    bpmSlider.setVisible (! proc.useDawBpm);
    bpmLabel.setVisible (! proc.useDawBpm);

    seedSlider.setVisible (! randomSeedToggle.getToggleState());

    bool fx = proc.fxEnabled;
    compThresholdSlider.setVisible (fx);
    compRatioSlider.setVisible (fx);
    delayTimeSlider.setVisible (fx);
    delayMixSlider.setVisible (fx);
    reverbSizeSlider.setVisible (fx);
    reverbMixSlider.setVisible (fx);

    resized();
}

void MLXAudioGenEditor::timerCallback()
{
    displayProgress = proc.getProgress();

    bool gen = proc.isGenerating();
    generateButton.setEnabled (! gen);
    generateButton.setButtonText (gen
        ? juce::String ("Generating ") + juce::String ((int) (displayProgress * 100)) + "%"
        : "Generate");

    playButton.setEnabled (proc.hasAudioLoaded());
    stopButton.setEnabled (proc.hasAudioLoaded());
    playButton.setButtonText (proc.isPlaying() ? "Pause" : "Play");

    statusLabel.setText (proc.getStatusMessage(), juce::dontSendNotification);

    // BPM display
    float bpm = proc.getEffectiveBpm();
    bpmDisplay.setText (juce::String ((int) bpm) + " BPM", juce::dontSendNotification);

    auto err = proc.getLastError();
    errorLabel.setText (err, juce::dontSendNotification);
    errorLabel.setVisible (err.isNotEmpty());

    repaint();
}

void MLXAudioGenEditor::onGenerateClicked()
{
    proc.triggerGeneration();
}
