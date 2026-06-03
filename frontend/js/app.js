function appState() {
            return {
                dashboardState: 'loading',
                currentPage: 'dashboard',
                pageContent: '<div style="padding: 20px;"><i class="fas fa-spinner fa-spin"></i> Đang tải...</div>',
                sidebarOpen: false,
                stats: {
                    cpu: 0,
                    memory: 0,
                    memory_used_gb: 0,
                    memory_total_gb: 0,
                    uptime: '-',
                    gpu: 'N/A',
                    gpu_name: null,
                    gpu_vram_gb: null,
                    gpu_temp_c: null,
                    gpu_power_w: null,
                    gpu_vram_used_mb: null,
                    gpu_utilization: null,
                    unified_memory: false,
                    models_count: 0,
                    datasets_count: 0,
                    training_count: 0,
                    tests_count: 0
                },
                models: [],
                datasets: [],
                activities: [],
                selectedModel: '',
                testInput: '',
                testChatMessages: [],
                testTemperature: 0.7,
                testTopP: 0.9,
                testTopK: 40,
                testRepPenalty: 1.1,
                testMaxTokens: 2048,
                currentBenchmark: { ttft: null, tps: null, totalTime: null, tokens: null },
                localInferenceStats: { queue_length: 0, last_tps: null, last_ttft: null, total_requests: 0 },
                
                // --- Terminal & Logs ---
                terminalLines: [],
                inferenceHistory: [],
                isTesting: false,
                abortController: null,
                modelStatus: { enabled: false, base_url: null, models: [] },
                systemInfo: { project: 'Synapse AI', version: '-' },
                networkNodes: [],
                distributedStatus: { partitions_count: 0, multi_node_inference: false, multi_node_training: false, message: '' },
                modelList: { downloaded: [], available: [] },
                pullLoading: '',
                deleteLoading: '',
                trainingModel: '',
                trainingDataset: '',
                trainingEpochs: 10,
                trainingBatchSize: 16,
                trainingLearningRate: '1e-5',
                trainingSaveEvery: 5,
                trainingMaxSteps: 0,
                trainingWarmupSteps: 0,
                trainingOutputName: '',
                trainingCheckpointDir: 'checkpoints',
                trainingSystemPrompt: '',
                trainingMaxLength: 128,
                trainingStatus: 'idle',
                trainingJob: null,
                trainingLoading: false,
                uploadLoading: false,
                deleteDatasetLoading: '',
                datasetPreview: [],
                datasetPreviewName: '',
                datasetPreviewLines: 50,
                datasetPreviewTarget: null,
                datasetCreateOpen: false,
                datasetCreateName: '',
                datasetCreateFormat: 'jsonl',
                datasetCreateTemplate: 'empty',
                datasetCreateContent: '',
                datasetCreateLoading: false,
                datasetRenameOpen: false,
                datasetRenameTarget: null,
                datasetRenameNewName: '',
                datasetRenameLoading: false,
                validateDatasetLoading: '',
                settingsForm: { default_model: '', theme: 'light', discovery_module: 'tailscale' },
                settingsLoading: false,
                pageLoading: false,
                datasetValidateResult: null,
                // === Cluster Resources ===
                clusterResources: [],
                clusterLoading: false,
                clusterTrainingJob: null,
                clusterStrategyName: '',
                clusterTotalLayers: 32,
                clusterPeerGraph: {},
                // Removed chart refs to avoid Alpine Proxy issues
                _lossHistory: [],
                _monitoringHistory: { labels: [], cpu: [], ram: [], vram: [] },
                // === Model filter ===
                modelSearch: '',
                modelStatusFilter: 'all',  // 'all', 'downloaded', 'available'
                
                async loadPage(page) {
                    this.pageLoading = true;
                    this.destroyAllCharts();
                    try {
                        let newContent = '';
                        const res = await fetch(`/_static/pages/${page}.html`);
                        if (res.ok) {
                            newContent = await res.text();
                        } else {
                            const resVite = await fetch(`/pages/${page}.html`);
                            if (resVite.ok) {
                                newContent = await resVite.text();
                            } else {
                                newContent = `<div class="text-danger">Lỗi tải trang ${page}</div>`;
                            }
                        }

                        if (newContent.includes('<html')) {
                            newContent = `<div class="text-danger">Lỗi tải trang ${page} (SPA Fallback)</div>`;
                        }

                        this.pageContent = newContent;
                        this.$nextTick(() => {
                            const container = document.getElementById('page-container');
                            if (container && window.Alpine) {
                                Alpine.initTree(container);
                            }
                        });
                    } catch (e) {
                        this.pageContent = `<div class="text-danger">Lỗi kết nối khi tải trang ${page}</div>`;
                    } finally {
                        this.pageLoading = false;
                    }
                },
                // Khởi tạo
                async init() {
                    // Cài đặt Fetch Interceptor cho Authentication
                    const originalFetch = window.fetch;
                    const self = this;
                    window.fetch = async function(...args) {
                        let [resource, config] = args;
                        if (self.settingsForm && self.settingsForm.api_auth_enabled === 'true' && self.settingsForm.api_key) {
                            config = config || {};
                            config.headers = config.headers || {};
                            if (config.headers instanceof Headers) {
                                config.headers.set('Authorization', 'Bearer ' + self.settingsForm.api_key);
                            } else {
                                config.headers['Authorization'] = 'Bearer ' + self.settingsForm.api_key;
                            }
                        }
                        return originalFetch.call(this, resource, config);
                    };

                    // Removed forced dark mode to allow white theme by default
                    this.$watch('currentPage', (val) => this.loadPage(val));
                    this.loadPage(this.currentPage);
                    this.fetchSystemInfo();
                    this.fetchStats();
                    this.fetchModelList();
                    this.fetchModelStatus();
                    this.fetchActivity();
                    this.fetchTerminalLogs();
                    this.fetchModelStatus();
                    this.fetchSystemInfo();
                    this.fetchNetworkNodes();
                    this.fetchDistributedStatus();
                    this.fetchTrainingStatus();
                    this.fetchClusterResources();
                    this.fetchDatasets();
                    this.fetchModels();

                    this.settingsForm.theme = 'light';
                    localStorage.setItem('synapse_theme', 'light');
                    this.applyTheme(this.settingsForm.theme);
                    
                    // Khởi tạo WebSocket thay vì dùng Polling
                    this.initWebSocket();
                    
                    // Thêm Watchdog Timer để chống "Silent Deadlock"
                    setInterval(() => {
                        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                            if (Date.now() - this.lastApiUpdate > 15000) {
                                console.warn("WebSocket data stale (>15s). Forcing reconnect...");
                                this.ws.close();
                            }
                        }
                    }, 5000);
                },
                
                // ===== WEBSOCKET =====
                ws: null,
                wsReconnectAttempts: 0,
                initWebSocket() {
                    if (this.ws) return;
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/v1/ws/metrics`;
                    
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        if (this.wsReconnectAttempts > 0) {
                            this.showToast('Đã kết nối lại với máy chủ', 'success');
                        }
                        this.wsReconnectAttempts = 0;
                        this.isOffline = false;
                        if (this.dashboardState === 'loading' || this.dashboardState === 'error') {
                            this.dashboardState = 'ready';
                        }
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'metrics') {
                                this.updateFromWebSocket(data);
                            }
                        } catch (e) {
                            console.error('WS Parse Error', e);
                        }
                    };
                    
                    this.ws.onclose = () => {
                        this.ws = null;
                        this.isOffline = true;
                        if (this.currentPage === 'dashboard') {
                            this.dashboardState = 'error';
                        }
                        if (this.wsReconnectAttempts === 0) {
                            this.showToast('Mất kết nối máy chủ. Đang thử lại...', 'error');
                        }
                        let delay = Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 10000);
                        this.wsReconnectAttempts++;
                        setTimeout(() => this.initWebSocket(), delay);
                    };
                    
                    this.ws.onerror = (err) => {
                        if (this.ws) this.ws.close();
                    };
                },
                
                showToast(message, type = 'info') {
                    const container = document.getElementById('toast-container');
                    if (!container) return;
                    const toast = document.createElement('div');
                    toast.className = `toast ${type}`;
                    
                    let icon = 'fa-info-circle';
                    if (type === 'success') icon = 'fa-check-circle';
                    if (type === 'error') icon = 'fa-exclamation-circle';
                    if (type === 'warning') icon = 'fa-exclamation-triangle';

                    toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
                    container.appendChild(toast);
                    
                    setTimeout(() => {
                        toast.classList.add('fade-out');
                        setTimeout(() => toast.remove(), 300);
                    }, 3000);
                },
                
                updateFromWebSocket(data) {
                    this.lastApiUpdate = Date.now();
                    
                    if (data.stats) {
                        this.stats = { ...this.stats, ...data.stats };
                        this.stats.models_count = data.stats.models_count ?? this.models.length;
                        
                        const maxPoints = 30;
                        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                        this._monitoringHistory.labels.push(now);
                        this._monitoringHistory.cpu.push(this.stats.cpu || 0);
                        this._monitoringHistory.ram.push(this.stats.memory || 0);
                        this._monitoringHistory.vram.push(this.stats.gpu_vram_used_mb ? this.stats.gpu_vram_used_mb / 1024 : 0);
                        
                        if (this._monitoringHistory.labels.length > maxPoints) {
                            this._monitoringHistory.labels.shift();
                            this._monitoringHistory.cpu.shift();
                            this._monitoringHistory.ram.shift();
                            this._monitoringHistory.vram.shift();
                        }
                        
                        if (this.currentPage === 'dashboard') {
                            this.renderSparklines();
                        } else if (this.currentPage === 'monitoring') {
                            this.renderMonitoringCharts();
                        }
                    }
                    
                    if (data.training !== undefined) {
                        this.trainingJob = data.training;
                        if (this.currentPage === 'training' && this.trainingJob) {
                            this.renderLossChart();
                        }
                    }
                    
                    if (data.terminal_logs !== undefined && this.currentPage === 'dashboard') {
                        if (JSON.stringify(this.terminalLogs) !== JSON.stringify(data.terminal_logs)) {
                            this.terminalLogs = data.terminal_logs;
                            this.$nextTick(() => {
                                const container = document.getElementById('terminalLogContainer');
                                if (container) container.scrollTop = container.scrollHeight;
                            });
                        }
                    }
                    
                    if (data.nodes) {
                        this.networkNodes = data.nodes;
                    }
                    
                    if (data.distributed) {
                        this.clusterTotalNodes = data.distributed.total_nodes || 0;
                        this.clusterOnlineNodes = data.distributed.nodes ? data.distributed.nodes.length : 0;
                        this.clusterStrategyName = data.distributed.strategy || 'Unknown';
                    }
                },
                applyTheme(theme) {
                    if (!theme) return;
                    localStorage.setItem('synapse_theme', theme);
                    document.documentElement.classList.toggle('theme-dark', theme === 'dark');
                    document.body.classList.toggle('theme-dark', theme === 'dark');
                },
                async fetchSettings() {
                    try {
                        const res = await fetch('/v1/settings');
                        const data = await res.json();
                        this.settingsForm.default_model = data.default_model || '';
                        this.settingsForm.discovery_module = data.discovery_module || 'tailscale';
                    } catch (e) {
                        console.error('fetchSettings:', e);
                    }
                },
                async applyDefaultModelForTest() {
                    await this.fetchSettings();
                    if (this.settingsForm.default_model) {
                        this.selectedModel = this.settingsForm.default_model;
                    }
                },
                async applyDefaultModelForTraining() {
                    await this.fetchSettings();
                    if (this.settingsForm.default_model) {
                        this.trainingModel = this.settingsForm.default_model;
                    }
                },
                async saveSettings() {
                    this.settingsLoading = true;
                    try {
                        const res = await fetch('/v1/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                default_model: this.settingsForm.default_model || undefined,
                                discovery_module: this.settingsForm.discovery_module || 'tailscale'
                            })
                        });
                        const data = await res.json();
                        if (data.success) {
                            await this.fetchModelStatus();
                            await this.fetchModelList();
                            await this.fetchActivity();
                            if (data.restarted) {
                                alert("Cài đặt đã được lưu. Hệ thống đang tự động khởi động lại mạng dưới nền để áp dụng phương thức kết nối mới. Vui lòng đợi trong giây lát...");
                                this.isOffline = true;
                                setTimeout(() => { this.fetchSystemInfo(); }, 3000);
                            } else {
                                alert("Cài đặt đã được lưu thành công.");
                            }
                        } else {
                            alert(data.message || 'Lưu thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.settingsLoading = false;
                    }
                },

                async startTraining() {
                    if (!this.trainingModel || !this.trainingDataset || this.trainingLoading) return;
                    this.trainingLoading = true;
                    try {
                        const res = await fetch('/v1/training/start', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                model: this.trainingModel,
                                dataset: this.trainingDataset,
                                epochs: this.trainingEpochs,
                                batch_size: this.trainingBatchSize,
                                learning_rate: this.trainingLearningRate || undefined,
                                save_every: this.trainingSaveEvery,
                                max_steps: this.trainingMaxSteps || 0,
                                warmup_steps: this.trainingWarmupSteps || 0,
                                max_length: this.trainingMaxLength || undefined,
                                output_model_name: this.trainingOutputName || undefined,
                                checkpoint_dir: this.trainingCheckpointDir || undefined,
                                system_prompt: this.trainingSystemPrompt || undefined
                            })
                        });
                        const data = await res.json();
                        if (data.success) {
                            await this.fetchTrainingStatus();
                            await this.fetchActivity();
                        } else {
                            alert(data.message || 'Khởi chạy training thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.trainingLoading = false;
                    }
                },

                async fetchTrainingStatus() {
                    try {
                        const res = await fetch('/v1/training/status');
                        const data = await res.json();
                        this.trainingStatus = data.status || 'idle';
                        const prevStatus = this.trainingJob && this.trainingJob.status;
                        this.trainingJob = data.job || null;
                        if (this.trainingJob && this.trainingJob.status === 'completed' && prevStatus !== 'completed') {
                            await this.fetchModelList();
                            await this.fetchModels();
                        }
                        
                        if (this.trainingJob && this.trainingJob.status === 'idle') {
                            this._lossHistory = [];
                        }
                        this.$nextTick(() => {
                            if (this.currentPage === 'training') this.renderLossChart();
                        });
                    } catch (e) {
                        this.trainingStatus = 'idle';
                        this.trainingJob = null;
                    }
                },

                async fetchTerminalLogs() {
                    try {
                        const res = await fetch('/v1/logs/terminal');
                        const data = await res.json();
                        this.terminalLines = data.data || [];
                    } catch (e) {
                        this.terminalLines = [];
                    }
                },

                async uploadDatasetFile(ev) {
                    const file = ev.target.files && ev.target.files[0];
                    if (!file) return;
                    this.uploadLoading = true;
                    try {
                        const form = new FormData();
                        form.append('file', file);
                        const res = await fetch('/v1/datasets/upload', { method: 'POST', body: form });
                        const data = await res.json();
                        if (data.success) {
                            await this.fetchDatasets();
                            await this.fetchStats();
                        } else {
                            alert(data.message || 'Upload thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.uploadLoading = false;
                        ev.target.value = '';
                    }
                },

                async viewDataset(dataset) {
                    this.datasetPreviewTarget = dataset;
                    const lines = this.datasetPreviewLines || 50;
                    try {
                        const res = await fetch('/v1/datasets/preview?path=' + encodeURIComponent(dataset.path) + '&lines=' + lines);
                        const data = await res.json();
                        if (data.success) {
                            this.datasetPreview = data.preview || [];
                            this.datasetPreviewName = dataset.name;
                        } else {
                            alert(data.message || 'Không xem được');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    }
                },

                downloadDataset(dataset) {
                    window.location.href = '/v1/datasets/download?path=' + encodeURIComponent(dataset.path);
                },

                openRenameDataset(dataset) {
                    this.datasetRenameTarget = dataset;
                    this.datasetRenameNewName = dataset.name;
                    this.datasetRenameOpen = true;
                },

                async renameDataset() {
                    if (!this.datasetRenameTarget || !this.datasetRenameNewName.trim() || this.datasetRenameLoading) return;
                    this.datasetRenameLoading = true;
                    try {
                        const res = await fetch('/v1/datasets/rename', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ name: this.datasetRenameTarget.name, new_name: this.datasetRenameNewName.trim() })
                        });
                        const data = await res.json();
                        if (data.success) {
                            if (this.trainingDataset === this.datasetRenameTarget.name) this.trainingDataset = data.name;
                            this.datasetRenameOpen = false;
                            this.datasetRenameTarget = null;
                            this.datasetRenameNewName = '';
                            await this.fetchDatasets();
                        } else {
                            alert(data.message || 'Đổi tên thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.datasetRenameLoading = false;
                    }
                },

                async validateDataset(dataset) {
                    this.validateDatasetLoading = dataset.name;
                    this.datasetValidateResult = null;
                    try {
                        const res = await fetch('/v1/datasets/validate?path=' + encodeURIComponent(dataset.path));
                        const data = await res.json();
                        if (data.success) {
                            this.datasetValidateResult = { valid_count: data.valid_count, total_lines: data.total_lines, format_detected: data.format_detected, errors: data.errors || [] };
                        } else {
                            alert(data.message || 'Kiểm tra thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.validateDatasetLoading = '';
                    }
                },

                async createDataset() {
                    const name = this.datasetCreateName.trim();
                    if (!name || this.datasetCreateLoading) return;
                    if (!name.endsWith('.json') && !name.endsWith('.jsonl')) {
                        alert('Tên file phải có đuôi .json hoặc .jsonl');
                        return;
                    }
                    this.datasetCreateLoading = true;
                    try {
                        const payload = { name: name, format: this.datasetCreateFormat, template: this.datasetCreateTemplate };
                        if (this.datasetCreateContent && this.datasetCreateContent.trim()) payload.content = this.datasetCreateContent.trim();
                        const res = await fetch('/v1/datasets/create', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        const data = await res.json();
                        if (data.success) {
                            this.datasetCreateOpen = false;
                            this.datasetCreateName = '';
                            this.datasetCreateContent = '';
                            await this.fetchDatasets();
                            await this.fetchStats();
                        } else {
                            alert(data.message || 'Tạo dataset thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.datasetCreateLoading = false;
                    }
                },

                async deleteDataset(dataset) {
                    if (!confirm('Xóa dataset "' + dataset.name + '"?')) return;
                    this.deleteDatasetLoading = dataset.name;
                    try {
                        const res = await fetch('/v1/datasets/delete', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ name: dataset.name })
                        });
                        const data = await res.json();
                        if (data.success) {
                            this.datasetPreview = [];
                            this.datasetPreviewName = '';
                            await this.fetchDatasets();
                            await this.fetchStats();
                        } else {
                            alert(data.message || 'Xóa thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.deleteDatasetLoading = '';
                    }
                },

                async fetchModelList() {
                try {
                const res = await fetch('/v1/models/list');
                const data = await res.json();
                if (!res.ok) throw new Error(data?.message || 'Lỗi ' + res.status);
                this.modelList = {
                downloaded: Array.isArray(data.downloaded) ? data.downloaded : [],
                available: Array.isArray(data.available) ? data.available : []
                };
                // Gọi API compatibility để lấy thông tin fit, memory_needed_gb, estimated_speed
                try {
                const compatRes = await fetch('/v1/models/compatibility');
                if (compatRes.ok) {
                const compatData = await compatRes.json();
                const compatMap = {};
                if (Array.isArray(compatData.models)) {
                compatData.models.forEach(m => { compatMap[m.name] = m; });
                }
                // Merge compatibility vào downloaded
                this.modelList.downloaded = this.modelList.downloaded.map(m => ({
                ...m,
                fit: compatMap[m.name]?.fit || null,
                memory_needed_gb: compatMap[m.name]?.memory_needed_gb || null,
                estimated_speed: compatMap[m.name]?.estimated_speed || null
                }));
                // Merge compatibility vào available
                this.modelList.available = this.modelList.available.map(m => ({
                ...m,
                fit: compatMap[m.name]?.fit || null,
                memory_needed_gb: compatMap[m.name]?.memory_needed_gb || null,
                estimated_speed: compatMap[m.name]?.estimated_speed || null
                }));
                }
                } catch (e2) { console.warn('fetchModelList compatibility:', e2); }
                this.models = (this.modelList.downloaded || []).map(m => ({ id: m.name, object: 'model', owned_by: 'pytorch', ready: true }));
                this.stats.models_count = this.models.length;
                if (this.models.length > 0 && !this.selectedModel) {
                this.selectedModel = this.settingsForm.default_model && this.modelList.downloaded.some(m => m.name === this.settingsForm.default_model)
                ? this.settingsForm.default_model
                : this.models[0].id;
                }
                } catch (e) {
                console.error('fetchModelList:', e);
                this.modelList = { downloaded: [], available: [] };
                }
                },

                async pullModel(modelName) {
                    if (!modelName || this.pullLoading) return;
                    this.pullLoading = modelName;
                    try {
                        const res = await fetch('/v1/models/pull', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: modelName }) });
                        const data = await res.json();
                        if (data.success) {
                            await this.fetchModelList();
                            await this.fetchModels();
                        } else {
                            alert(data.message || 'Tải thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.pullLoading = '';
                    }
                },

                async deleteModel(modelName) {
                    if (!modelName || this.deleteLoading) return;
                    if (!confirm('Xóa model "' + modelName + '"? Không thể hoàn tác.')) return;
                    this.deleteLoading = modelName;
                    try {
                        const res = await fetch('/v1/models/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: modelName }) });
                        const data = await res.json();
                        if (data.success) {
                            await this.fetchModelList();
                            await this.fetchModels();
                        } else {
                            alert(data.message || 'Xóa thất bại');
                        }
                    } catch (e) {
                        alert('Lỗi: ' + (e.message || e));
                    } finally {
                        this.deleteLoading = '';
                    }
                },

                async fetchNetworkNodes() {
                    try {
                        const res = await fetch('/v1/network/nodes');
                        const data = await res.json();
                        this.networkNodes = data.data || [];
                    } catch (e) {
                        this.networkNodes = [];
                    }
                },

                async fetchDistributedStatus() {
                    try {
                        const res = await fetch('/v1/distributed/status');
                        const data = await res.json();
                        this.distributedStatus = {
                            partitions_count: data.partitions_count ?? 0,
                            multi_node_inference: !!data.multi_node_inference,
                            multi_node_training: !!data.multi_node_training,
                            message: data.message || ''
                        };
                    } catch (e) {
                        this.distributedStatus = { partitions_count: 0, multi_node_inference: false, multi_node_training: false, message: '' };
                    }
                },

                async fetchModelStatus() {
                    try {
                        const res = await fetch('/v1/models/status');
                        const data = await res.json();
                        this.modelStatus = {
                            enabled: !!data.enabled,
                            base_url: data.base_url || null,
                            models: data.models || []
                        };
                    } catch (e) {
                        this.modelStatus = { enabled: false, base_url: null, models: [] };
                    }
                },

                async fetchSystemInfo() {
                try {
                const res = await fetch('/v1/system/info');
                const data = await res.json();
                this.systemInfo = {
                project: data.project || 'Synapse AI',
                version: data.version || '-',
                hardware: data.hardware || {},
                network_discovery: data.network_discovery || 'Unknown'
                };
                } catch (e) {
                this.systemInfo = { project: 'Synapse AI', version: '-', hardware: {}, network_discovery: 'Unknown' };
                }
                },
                
                isOffline: false,
                lastApiUpdate: 0,
                async fetchStats() {
                    try {
                        const res = await fetch('/v1/system/stats');
                        if (!res.ok) throw new Error('API Error');
                        const data = await res.json();
                        
                        this.lastApiUpdate = Date.now();
                        this.isOffline = false;
                        
                        if (this.dashboardState === 'loading' || this.dashboardState === 'error') {
                            this.dashboardState = 'ready';
                        }
                        
                        this.stats.cpu = data.cpu ?? 0;
                        this.stats.memory = data.memory ?? 0;
                        this.stats.memory_used_gb = data.memory_used_gb ?? 0;
                        this.stats.memory_total_gb = data.memory_total_gb ?? 0;
                        this.stats.uptime = data.uptime ?? '-';
                        this.stats.gpu = data.gpu ?? 'N/A';
                        this.stats.gpu_name = data.gpu_name ?? null;
                        this.stats.gpu_vram_gb = data.gpu_vram_gb ?? null;
                        this.stats.unified_memory = data.unified_memory ?? false;
                        this.stats.gpu_utilization = data.gpu_utilization ?? null;
                        this.stats.gpu_vram_used_mb = data.gpu_vram_used_mb ?? null;
                        this.stats.gpu_power_w = data.gpu_power_w ?? null;
                        this.stats.gpu_temp_c = data.gpu_temp_c ?? null;
                        this.stats.models_count = data.models_count ?? this.models.length;
                        this.stats.datasets_count = data.datasets_count ?? this.datasets.length;
                        
                        const maxPoints = 30; // 30 points * 5s = 150s history
                        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                        this._monitoringHistory.labels.push(now);
                        this._monitoringHistory.cpu.push(this.stats.cpu || 0);
                        this._monitoringHistory.ram.push(this.stats.memory || 0);
                        this._monitoringHistory.vram.push(this.stats.gpu_vram_used_mb ? this.stats.gpu_vram_used_mb / 1024 : 0);

                        if (this._monitoringHistory.labels.length > maxPoints) {
                            this._monitoringHistory.labels.shift();
                            this._monitoringHistory.cpu.shift();
                            this._monitoringHistory.ram.shift();
                            this._monitoringHistory.vram.shift();
                        }

                        this.$nextTick(() => {
                            if (this.currentPage === 'monitoring') this.renderMonitoringCharts();
                            if (this.currentPage === 'dashboard') this.renderDashboardSparklines();
                        });
                    } catch (e) {
                        if (Date.now() - this.lastApiUpdate > 10000) {
                            this.isOffline = true;
                            this.dashboardState = 'error';
                        } else if (this.dashboardState === 'loading') {
                            this.dashboardState = 'error';
                        }
                        console.error('Error fetching stats:', e);
                    }
                },

                renderDashboardSparklines() {
                    if (typeof Chart === 'undefined') return;
                    setTimeout(() => {
                        const createSparkline = (canvasId, dataColor, dataArr, maxVal) => {
                            const canvas = document.getElementById(canvasId);
                            if (!canvas) return;
                            let chartRef = Chart.getChart(canvas);
                            if (chartRef) {
                                chartRef.data.labels = this._monitoringHistory.labels;
                                chartRef.data.datasets[0].data = dataArr;
                                chartRef.update('none');
                            } else {
                                const ctx = canvas.getContext('2d');
                                // Create gradient
                                let gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                                gradient.addColorStop(0, dataColor + '40'); // 25% opacity
                                gradient.addColorStop(1, dataColor + '00'); // 0% opacity

                                new Chart(canvas, {
                                    type: 'line',
                                    data: {
                                        labels: this._monitoringHistory.labels,
                                        datasets: [{
                                            data: dataArr,
                                            borderColor: dataColor,
                                            backgroundColor: gradient,
                                            borderWidth: 1.5,
                                            pointRadius: 0,
                                            fill: true,
                                            tension: 0.3
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        animation: false,
                                        plugins: { legend: { display: false }, tooltip: { enabled: false } },
                                        scales: {
                                            x: { display: false },
                                            y: { display: false, min: 0, max: maxVal }
                                        },
                                        layout: { padding: 0 }
                                    }
                                });
                            }
                        };

                        createSparkline('dashCpuSparkline', '#38BDF8', this._monitoringHistory.cpu, 100);
                        const vramMax = this.stats.gpu_vram_gb ? this.stats.gpu_vram_gb : 24;
                        createSparkline('dashVramSparkline', '#8B5CF6', this._monitoringHistory.vram, vramMax);
                    }, 50);
                },

                renderMonitoringCharts() {
                    if (typeof Chart === 'undefined') return;
                    setTimeout(() => {

                    const updateOrCreate = (canvasId, label, dataColor, dataArr, maxVal) => {
                        const canvas = document.getElementById(canvasId);
                        if (!canvas) return;
                        let chartRef = Chart.getChart(canvas);
                        if (chartRef) {
                            chartRef.data.labels = this._monitoringHistory.labels;
                            chartRef.data.datasets[0].data = dataArr;
                            chartRef.update('none');
                        } else {
                            new Chart(canvas, {
                                type: 'line',
                                data: {
                                    labels: this._monitoringHistory.labels,
                                    datasets: [{
                                        label: label,
                                        data: dataArr,
                                        borderColor: dataColor,
                                        backgroundColor: dataColor + '33',
                                        borderWidth: 2,
                                        fill: true,
                                        tension: 0.3,
                                        pointRadius: 0
                                    }]
                                },
                                options: {
                                    responsive: true, maintainAspectRatio: false,
                                    animation: { duration: 0 },
                                    plugins: { legend: { display: false } },
                                    scales: {
                                        x: { display: false },
                                        y: { min: 0, max: maxVal, border: { dash: [4, 4] }, grid: { color: 'rgba(0,0,0,0.05)' } }
                                    }
                                }
                            });
                        }
                    };

                    updateOrCreate('cpuChart', 'CPU (%)', '#0EA5E9', this._monitoringHistory.cpu, 100);
                    updateOrCreate('ramChart', 'RAM (%)', '#10B981', this._monitoringHistory.ram, 100);
                    const vramMax = this.stats.gpu_vram_gb ? this.stats.gpu_vram_gb : 24;
                    updateOrCreate('vramChart', 'VRAM (GB)', '#8B5CF6', this._monitoringHistory.vram, vramMax);
                    }, 50);
                },

                renderLossChart() {
                    if (typeof Chart === 'undefined') return;
                    const canvas = document.getElementById('lossChart');
                    if (!canvas) return;

                    if (this.trainingJob && this.trainingJob.loss != null) {
                        const currentLoss = parseFloat(this.trainingJob.loss);
                        if (!isNaN(currentLoss)) {
                            // Only add if different or at a new time step
                            this._lossHistory.push({ x: this._lossHistory.length, y: currentLoss });
                        }
                    }

                    let chartRef = Chart.getChart(canvas);
                    if (chartRef) {
                        chartRef.data.datasets[0].data = this._lossHistory;
                        chartRef.update('none');
                    } else {
                        new Chart(canvas, {
                            type: 'line',
                            data: {
                                datasets: [{
                                    label: 'Training Loss',
                                    data: this._lossHistory,
                                    borderColor: '#EF4444',
                                    borderWidth: 2,
                                    fill: false,
                                    tension: 0.2,
                                    pointRadius: 2
                                }]
                            },
                            options: {
                                responsive: true, maintainAspectRatio: false,
                                plugins: { legend: { display: false } },
                                scales: {
                                    x: { type: 'linear', title: { display: true, text: 'Steps' } },
                                    y: { title: { display: true, text: 'Loss' } }
                                }
                            }
                        });
                    }
                },

                async fetchModels() {
                    try {
                        const res = await fetch('/v1/models');
                        const data = await res.json();
                        this.models = data.data || [];
                        if (this.models.length > 0 && !this.selectedModel) {
                            const defaultId = this.settingsForm.default_model;
                            const hasDefault = defaultId && this.models.some(m => (m.id || m.name) === defaultId);
                            this.selectedModel = hasDefault ? defaultId : this.models[0].id;
                        }
                    } catch (e) {
                        console.error('Error fetching models:', e);
                    }
                },

                async fetchDatasets() {
                    try {
                        const res = await fetch('/v1/datasets');
                        const data = await res.json();
                        this.datasets = data.data || [];
                        this.stats.datasets_count = this.datasets.length;
                    } catch (e) {
                        console.error('Error fetching datasets:', e);
                    }
                },

                async fetchActivity() {
                    try {
                        const res = await fetch('/v1/activity');
                        const data = await res.json();
                        this.activities = data.data || [];
                    } catch (e) {
                        console.error('Error fetching activity:', e);
                    }
                },

                async runTest() {
                    if (!this.selectedModel || !this.testInput.trim()) return;
                    const userContent = this.testInput.trim();
                    this.testInput = '';
                    this.testChatMessages.push({ role: 'user', content: userContent });
                    this.isTesting = true;

                    // Ép browser vẽ lại (spinner quay, nút đỏ) trước khi gọi API
                    await new Promise(r => requestAnimationFrame(r));
                    await new Promise(r => requestAnimationFrame(r));

                    try {
                        const temp = typeof this.testTemperature === 'number' ? this.testTemperature : 0.7;
                        const messages = this.testChatMessages.map(m => ({ role: m.role, content: m.content }));

                        const startTime = performance.now();
                        this.abortController = new AbortController();
                        
                        this.localInferenceStats.queue_length++;

                        const res = await fetch('/v1/chat/completions', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                model: this.selectedModel,
                                messages: messages,
                                temperature: temp,
                                top_p: this.testTopP,
                                top_k: this.testTopK,
                                repetition_penalty: this.testRepPenalty,
                                max_tokens: this.testMaxTokens || undefined,
                                stream: false
                            }),
                            signal: this.abortController.signal
                        });

                        // Tính thời gian đã trôi qua
                        const endTime = performance.now();
                        const elapsedSeconds = (endTime - startTime) / 1000;
                        this.currentBenchmark.totalTime = elapsedSeconds;

                        const data = await res.json();
                        if (!res.ok) {
                            this.testChatMessages.push({ role: 'assistant', content: 'Lỗi ' + res.status + ': ' + (data.detail || data.error?.message || data.message || JSON.stringify(data)), error: true });
                            return;
                        }
                        if (data.error) {
                            this.testChatMessages.push({ role: 'assistant', content: 'Lỗi: ' + (data.error.message || data.error.detail || JSON.stringify(data.error)), error: true });
                            return;
                        }
                        if (data.choices && data.choices[0]) {
                            const content = data.choices[0].message?.content || '(Không có nội dung)';
                            let usage = '';
                            if (data.usage) {
                                const u = data.usage;
                                const completionTokens = u.completion_tokens || 0;
                                const promptTokens = u.prompt_tokens || 0;
                                const totalTokens = u.total_tokens || 0;

                                // Tính token/s
                                let tokensPerSecond = '';
                                if (completionTokens > 0 && elapsedSeconds > 0) {
                                    const tps = (completionTokens / elapsedSeconds);
                                    this.currentBenchmark.tps = tps;
                                    this.localInferenceStats.last_tps = tps;
                                    tokensPerSecond = ` · ${tps.toFixed(2)} token/s`;
                                }
                                
                                // Mô phỏng TTFT bằng 1/3 tổng thời gian (vì non-streaming không có TTFT chuẩn)
                                const ttft = elapsedSeconds * 0.3;
                                this.currentBenchmark.ttft = ttft;
                                this.localInferenceStats.last_ttft = ttft;
                                this.currentBenchmark.tokens = completionTokens;

                                usage = `Token: đầu vào ${promptTokens}, đầu ra ${completionTokens}, tổng ${totalTokens}${tokensPerSecond}`;
                            }
                            this.testChatMessages.push({ role: 'assistant', content: content, usage: usage });
                            this.localInferenceStats.total_requests++;
                            
                            // Lưu vào history
                            this.inferenceHistory.unshift({
                                id: Math.random().toString(36).substr(2, 8),
                                time: new Date().toLocaleTimeString(),
                                prompt: userContent.substring(0, 50) + (userContent.length > 50 ? '...' : ''),
                                status: 'success',
                                latency: elapsedSeconds.toFixed(2) + 's',
                                tokens: data.usage?.completion_tokens || 0
                            });
                            if (this.inferenceHistory.length > 20) this.inferenceHistory.pop();
                            
                        } else {
                            this.testChatMessages.push({ role: 'assistant', content: 'Phản hồi không hợp lệ.', error: true });
                        }
                    } catch (e) {
                        // Nếu là AbortError thì không hiển thị lỗi (user đã cancel)
                        if (e.name !== 'AbortError') {
                            this.testChatMessages.push({ role: 'assistant', content: 'Lỗi kết nối: ' + (e.message || String(e)), error: true });
                        } else {
                            // Nếu đã cancel, thêm message "Đã dừng"
                            if (this.testChatMessages.length > 0 && this.testChatMessages[this.testChatMessages.length - 1].role === 'user') {
                                this.testChatMessages.push({ role: 'assistant', content: '(Đã dừng)', usage: '' });
                            }
                        }
                    } finally {
                        this.isTesting = false;
                        this.abortController = null;
                        if (this.localInferenceStats.queue_length > 0) this.localInferenceStats.queue_length--;
                        this.$nextTick(() => this.scrollChatToBottom());
                    }
                },

                cancelTest() {
                    if (this.abortController) {
                        this.abortController.abort();
                        this.abortController = null;
                    }
                    this.isTesting = false;
                },
                scrollChatToBottom() {
                    const el = document.getElementById('chat-messages');
                    if (el) el.scrollTop = el.scrollHeight;
                },

                getPageTitle() {
                    const titles = {
                        'dashboard': 'Dashboard',
                        'models': 'Quản Lý Mô Hình',
                        'test': 'Kiểm thử',
                        'training': 'Training',
                        'datasets': 'Quản Lý Dataset',
                        'monitoring': 'Giám Sát Hệ Thống',
                        'settings': 'Cài Đặt',
                        'cluster': 'Cluster Resources',
                        'about': 'Giới thiệu'
                    };
                    return titles[this.currentPage] || 'Synapse AI';
                },

                // ===== CHARTS & MONITORING =====
                destroyAllCharts() {
                    if (typeof Chart !== 'undefined') {
                        Object.values(Chart.instances).forEach(chart => {
                            chart.destroy();
                        });
                    }
                },
                },

                // ===== CLUSTER RESOURCES =====
                async fetchClusterResources() {
                    this.clusterLoading = true;
                    try {
                        const res = await fetch('/v1/cluster/resources');
                        const data = await res.json();
                        this.clusterResources = data.nodes || [];
                        this.clusterStrategyName = data.strategy_name || 'Unknown';
                        this.clusterTotalLayers = data.total_layers || 32;
                        this.clusterTrainingJob = data.training_job || null;
                        this.clusterPeerGraph = data.peer_graph || {};
                        this.$nextTick(() => this.renderClusterCharts());
                    } catch (e) {
                        console.error('fetchClusterResources:', e);
                        this.clusterResources = [];
                    } finally {
                        this.clusterLoading = false;
                    }
                },

                renderClusterCharts() {
                    if (typeof Chart === 'undefined') return;
                    const nodes = this.clusterResources;
                    if (!nodes || nodes.length === 0) return;
                    const COLORS = ['#0F766E', '#0EA5E9', '#D97706', '#7C3AED', '#059669', '#DC2626', '#F59E0B'];

                    const layerCanvas = document.getElementById('layerChart');
                    if (layerCanvas) {
                        let chartRef = Chart.getChart(layerCanvas);
                        if (chartRef) { chartRef.destroy(); }
                        new Chart(layerCanvas, {
                            type: 'bar',
                            data: {
                                labels: nodes.map(n => n.name),
                                datasets: [{
                                    label: 'Số lượng layer',
                                    data: nodes.map(n => n.layer_count),
                                    backgroundColor: nodes.map((_, i) => COLORS[i % COLORS.length] + 'CC'),
                                    borderColor: nodes.map((_, i) => COLORS[i % COLORS.length]),
                                    borderWidth: 2, borderRadius: 6,
                                }]
                            },
                            options: {
                                indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                                plugins: {
                                    legend: { display: false },
                                    tooltip: {
                                        callbacks: {
                                            label: ctx => {
                                                const n = nodes[ctx.dataIndex];
                                                return [` Layers: ${n.layer_start} → ${n.layer_end} (${n.layer_count} layers)`, ` Tỷ lệ: ${n.share_pct}%`,
                                                n.flops_fp16 > 0 ? ` TFLOPS: ${n.flops_fp16.toFixed(2)}` : ' CPU only'];
                                            }
                                        }
                                    }
                                },
                                scales: {
                                    x: { 
                                        max: this.clusterTotalLayers, 
                                        title: { display: true, text: 'Số lượng Layer' }, 
                                        grid: { color: 'rgba(0,0,0,0.05)' },
                                        ticks: { stepSize: 4 }
                                    },
                                    y: { grid: { display: false } }
                                }
                            }
                        });
                    }

                    const flopsCanvas = document.getElementById('flopsDonut');
                    if (flopsCanvas) {
                        if (this._flopsChart) { this._flopsChart.destroy(); this._flopsChart = null; }
                        const totalFlops = nodes.reduce((s, n) => s + (n.flops_fp16 || 0), 0);
                        const useRam = totalFlops === 0;
                        const values = useRam ? nodes.map(n => n.ram_gb || 0) : nodes.map(n => n.flops_fp16 || 0);
                        const totalVal = values.reduce((s, v) => s + v, 0);
                        this._flopsChart = new Chart(flopsCanvas, {
                            type: 'doughnut',
                            data: {
                                labels: nodes.map(n => n.name),
                                datasets: [{
                                    data: values,
                                    backgroundColor: nodes.map((_, i) => COLORS[i % COLORS.length] + 'CC'),
                                    borderColor: nodes.map((_, i) => COLORS[i % COLORS.length]), borderWidth: 2
                                }]
                            },
                            options: {
                                responsive: true, maintainAspectRatio: false, cutout: '60%',
                                plugins: {
                                    legend: { position: 'bottom', labels: { font: { size: 12 }, padding: 12 } },
                                    tooltip: {
                                        callbacks: {
                                            label: ctx => {
                                                const n = nodes[ctx.dataIndex];
                                                const val = values[ctx.dataIndex];
                                                const pct = totalVal > 0 ? (val / totalVal * 100).toFixed(1) : 0;
                                                return useRam ? ` ${n.ram_gb} GB RAM (${pct}%)` : ` ${n.flops_fp16.toFixed(2)} TFLOPS (${pct}%)`;
                                            }
                                        }
                                    }
                                }
                            }
                        });
                    }
                },

                allFilteredModels() {
                    const search = (this.modelSearch || '').toLowerCase();
                    const status = this.modelStatusFilter || 'all';
                    
                    // Gắn cờ cho từng model và map lại
                    const downloaded = (this.modelList.downloaded || []).map(m => ({ ...m, isDownloaded: true }));
                    const availableRaw = (this.modelList.available || []).map(m => ({ ...m, isDownloaded: false }));
                    
                    // Loại bỏ các model "Có thể tải" nếu nó đã nằm trong danh sách "Đã tải"
                    const downloadedNames = new Set(downloaded.map(m => m.name));
                    const available = availableRaw.filter(m => !downloadedNames.has(m.name));
                    
                    // Gộp 2 mảng lại: Đã tải sẽ luôn đứng trước (vì mảng downloaded đặt trước)
                    const all = [...downloaded, ...available];
                    
                    return all.filter(m => {
                        const matchSearch = !search || (m.name || '').toLowerCase().includes(search) || (m.family || '').toLowerCase().includes(search);
                        const matchStatus = status === 'all' 
                            || (status === 'downloaded' && m.isDownloaded) 
                            || (status === 'available' && !m.isDownloaded);
                        return matchSearch && matchStatus;
                    });
                },
                parseAnsi(text) {
                    if (!text) return '';
                    let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    const ansiRegex = /\x1b\[([\d;]+)m/g;
                    let spanOpen = false;
                    html = html.replace(ansiRegex, (match, codes) => {
                        const codeList = codes.split(';');
                        if (codeList.includes('0')) return spanOpen ? (spanOpen = false, '</span>') : '';
                        
                        let colorStyle = '';
                        const colors = { '31': 'color:#EF4444', '32': 'color:#22C55E', '33': 'color:#F59E0B', '34': 'color:#3B82F6', '36': 'color:#06B6D4' };
                        
                        for (let code of codeList) {
                            if (colors[code]) {
                                colorStyle = colors[code];
                            } else if (code === '1') {
                                colorStyle += (colorStyle ? ';' : '') + 'font-weight:bold';
                            }
                        }
                        
                        const prefix = spanOpen ? '</span>' : '';
                        spanOpen = !!colorStyle;
                        return colorStyle ? `${prefix}<span style="${colorStyle}">` : prefix;
                    });
                    return spanOpen ? html + '</span>' : html;
                }
            };
        }