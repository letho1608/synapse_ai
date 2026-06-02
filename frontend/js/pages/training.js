// Synapse AI — Training Page Logic
Synapse.Pages = Synapse.Pages || {};

Synapse.Pages.training = {
  async submitTraining() {
    const store = Alpine.store('app');
    const ui = Alpine.store('ui');
    if (!ui.trainingModel || !ui.trainingDataset) return;

    ui.setLoading('training', true);
    try {
      await store.startTraining({
        model: ui.trainingModel,
        dataset: ui.trainingDataset,
        epochs: ui.trainingEpochs,
        batch_size: ui.trainingBatchSize,
        learning_rate: ui.trainingLearningRate || undefined,
        save_every: ui.trainingSaveEvery,
        max_steps: ui.trainingMaxSteps || 0,
        warmup_steps: ui.trainingWarmupSteps || 0,
        max_length: ui.trainingMaxLength || undefined,
        output_model_name: ui.trainingOutputName || undefined,
        checkpoint_dir: ui.trainingCheckpointDir || undefined,
        system_prompt: ui.trainingSystemPrompt || undefined,
        // LoRA params (restored)
        lora_rank: ui.loraRank || 8,
        lora_alpha: ui.loraAlpha || 16,
        lora_dropout: ui.loraDropout || 0.1,
        lora_target_modules: ui.loraTargetModules || 'all'
      });
    } finally {
      ui.setLoading('training', false);
    }
  }
};