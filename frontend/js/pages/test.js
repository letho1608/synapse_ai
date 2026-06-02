// Synapse AI — Test/Chat Page Logic
Synapse.Pages = Synapse.Pages || {};

Synapse.Pages.test = {
  async sendMessage() {
    const store = Alpine.store('app');
    const ui = Alpine.store('ui');
    const model = ui.chatModel;
    const input = ui.chatInput.trim();
    if (!model || !input) return;

    ui.chatMessages.push({ role: 'user', content: input });
    ui.chatInput = '';
    ui.isChatting = true;
    ui.chatModel = ui.chatModel; // force reactivity

    await new Promise(r => requestAnimationFrame(r));
    await new Promise(r => requestAnimationFrame(r));

    try {
      const messages = ui.chatMessages.map(m => ({ role: m.role, content: m.content }));
      ui.abortController = new AbortController();
      const result = await store.trainChatCompletion(messages, ui.chatModel, ui.chatTemperature, ui.abortController.signal);
      ui.chatMessages.push({ role: 'assistant', content: result.content, usage: result.usage });
    } catch (e) {
      if (e.name !== 'AbortError') {
        ui.chatMessages.push({ role: 'assistant', content: 'Lỗi: ' + (e.message || String(e)), error: true });
      } else {
        const last = ui.chatMessages[ui.chatMessages.length - 1];
        if (last && last.role === 'user') {
          ui.chatMessages.push({ role: 'assistant', content: '(Đã dừng)', usage: '' });
        }
      }
    } finally {
      ui.isChatting = false;
      ui.abortController = null;
      setTimeout(() => {
        const el = document.getElementById('chat-messages');
        if (el) el.scrollTop = el.scrollHeight;
      }, 50);
    }
  }
};