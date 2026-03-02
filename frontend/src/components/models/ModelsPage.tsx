import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Cpu,
    Check,
    X,
    Zap,
    Globe,
    MessageSquare,
    Radio,
    ChevronRight,
    AlertCircle,
    Loader2,
    RefreshCw,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import Layout from '../layout/Layout';
import Button from '../ui/Button';

// API Base URL
const API_BASE = '';

interface ModelInfo {
    id: string;
    name: string;
    size: '1.7B' | '0.6B';
    type: 'VoiceDesign' | 'CustomVoice' | 'Base';
    description: string;
    features: {
        voiceDesign: boolean;
        customVoice: boolean;
        voiceClone: boolean;
        streaming: boolean;
        instructionControl: boolean;
    };
    languages: string[];
    memoryRequired: string;
}

const MODELS: ModelInfo[] = [
    {
        id: 'qwen3-tts-1.7b-voicedesign',
        name: 'Qwen3-TTS-12Hz-1.7B-VoiceDesign',
        size: '1.7B',
        type: 'VoiceDesign',
        description: 'Performs voice design based on user-provided descriptions. Create any voice you can imagine from text.',
        features: {
            voiceDesign: true,
            customVoice: false,
            voiceClone: false,
            streaming: true,
            instructionControl: true,
        },
        languages: ['Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'],
        memoryRequired: '~4GB VRAM',
    },
    {
        id: 'qwen3-tts-1.7b-customvoice',
        name: 'Qwen3-TTS-12Hz-1.7B-CustomVoice',
        size: '1.7B',
        type: 'CustomVoice',
        description: 'Style control over 9 premium timbres via user instructions. Covers various combinations of gender, age, language, and dialect.',
        features: {
            voiceDesign: false,
            customVoice: true,
            voiceClone: false,
            streaming: true,
            instructionControl: true,
        },
        languages: ['Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'],
        memoryRequired: '~4GB VRAM',
    },
    {
        id: 'qwen3-tts-1.7b-base',
        name: 'Qwen3-TTS-12Hz-1.7B-Base',
        size: '1.7B',
        type: 'Base',
        description: 'Base model capable of 3-second rapid voice clone from user audio input. Can be used for fine-tuning.',
        features: {
            voiceDesign: false,
            customVoice: false,
            voiceClone: true,
            streaming: true,
            instructionControl: false,
        },
        languages: ['Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'],
        memoryRequired: '~4GB VRAM',
    },
    {
        id: 'qwen3-tts-0.6b-customvoice',
        name: 'Qwen3-TTS-12Hz-0.6B-CustomVoice',
        size: '0.6B',
        type: 'CustomVoice',
        description: 'Lightweight model supporting 9 premium timbres. Faster inference with lower memory requirements.',
        features: {
            voiceDesign: false,
            customVoice: true,
            voiceClone: false,
            streaming: true,
            instructionControl: false,
        },
        languages: ['Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'],
        memoryRequired: '~2GB VRAM',
    },
    {
        id: 'qwen3-tts-0.6b-base',
        name: 'Qwen3-TTS-12Hz-0.6B-Base',
        size: '0.6B',
        type: 'Base',
        description: 'Lightweight base model for rapid voice cloning. Ideal for edge deployment and real-time applications.',
        features: {
            voiceDesign: false,
            customVoice: false,
            voiceClone: true,
            streaming: true,
            instructionControl: false,
        },
        languages: ['Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'],
        memoryRequired: '~2GB VRAM',
    },
];

interface CurrentConfig {
    model_size: string;
    enable_custom_voice: boolean;
    enable_voice_design: boolean;
    enable_voice_clone: boolean;
}

const ModelsPage: React.FC = () => {
    const [selectedSize, setSelectedSize] = useState<'1.7B' | '0.6B' | 'all'>('all');
    const [currentConfig, setCurrentConfig] = useState<CurrentConfig | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    // Fetch current backend config
    useEffect(() => {
        fetchCurrentConfig();
    }, []);

    const fetchCurrentConfig = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v1/qwen/speakers`);
            if (response.ok) {
                // The speakers endpoint tells us which backend is active
                setCurrentConfig({
                    model_size: '0.6B', // TODO: Get from actual backend config endpoint
                    enable_custom_voice: true,
                    enable_voice_design: false,
                    enable_voice_clone: true,
                });
            }
        } catch (error) {
            console.log('Could not fetch current config');
        }
    };

    const getTypeColor = (type: string) => {
        switch (type) {
            case 'VoiceDesign': return 'from-purple-500 to-pink-500';
            case 'CustomVoice': return 'from-blue-500 to-cyan-500';
            case 'Base': return 'from-green-500 to-emerald-500';
            default: return 'from-gray-500 to-gray-600';
        }
    };

    const getTypeBadgeColor = (type: string) => {
        switch (type) {
            case 'VoiceDesign': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
            case 'CustomVoice': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
            case 'Base': return 'bg-green-500/20 text-green-400 border-green-500/30';
            default: return 'bg-gray-500/20 text-gray-400';
        }
    };

    const filteredModels = selectedSize === 'all'
        ? MODELS
        : MODELS.filter(m => m.size === selectedSize);

    const FeatureIcon = ({ enabled }: { enabled: boolean }) => (
        enabled ? (
            <Check className="w-4 h-4 text-green-400" />
        ) : (
            <X className="w-4 h-4 text-text-tertiary opacity-50" />
        )
    );

    return (
        <Layout>
            <div className="h-full flex flex-col bg-background overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-border-default">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
                                <Cpu className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold text-text-primary">Models</h1>
                                <p className="text-sm text-text-secondary">Qwen3-TTS model configurations</p>
                            </div>
                        </div>

                        {/* Size Filter */}
                        <div className="flex items-center gap-2 bg-bg-secondary rounded-lg p-1">
                            {(['all', '1.7B', '0.6B'] as const).map((size) => (
                                <button
                                    key={size}
                                    onClick={() => setSelectedSize(size)}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${selectedSize === size
                                            ? 'bg-accent-primary text-white'
                                            : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                                        }`}
                                >
                                    {size === 'all' ? 'All Models' : size}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Current Configuration Banner */}
                {currentConfig && (
                    <div className="px-6 py-3 bg-accent-primary/10 border-b border-accent-primary/20">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                                <span className="text-sm text-text-secondary">
                                    Active Configuration: <span className="text-text-primary font-medium">{currentConfig.model_size}</span>
                                    {' • '}
                                    <span className="text-text-tertiary">
                                        CustomVoice: {currentConfig.enable_custom_voice ? 'On' : 'Off'} |
                                        VoiceDesign: {currentConfig.enable_voice_design ? 'On' : 'Off'} |
                                        VoiceClone: {currentConfig.enable_voice_clone ? 'On' : 'Off'}
                                    </span>
                                </span>
                            </div>
                            <Button
                                onClick={fetchCurrentConfig}
                                variant="ghost"
                                size="sm"
                                className="text-xs"
                            >
                                <RefreshCw className="w-3 h-3 mr-1" />
                                Refresh
                            </Button>
                        </div>
                    </div>
                )}

                {/* Models Grid */}
                <div className="flex-1 overflow-y-auto p-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {filteredModels.map((model, index) => (
                            <motion.div
                                key={model.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="group relative bg-bg-secondary rounded-2xl border border-border-default overflow-hidden hover:border-accent-primary/50 transition-all duration-300"
                            >
                                {/* Gradient accent */}
                                <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${getTypeColor(model.type)}`} />

                                <div className="p-6">
                                    {/* Header */}
                                    <div className="flex items-start justify-between mb-4">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-2">
                                                <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${getTypeBadgeColor(model.type)}`}>
                                                    {model.type}
                                                </span>
                                                <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${model.size === '1.7B'
                                                        ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                                                    }`}>
                                                    {model.size}
                                                </span>
                                            </div>
                                            <h3 className="text-lg font-semibold text-text-primary mb-1 font-mono text-sm">
                                                {model.name}
                                            </h3>
                                        </div>
                                    </div>

                                    {/* Description */}
                                    <p className="text-sm text-text-secondary mb-5 leading-relaxed">
                                        {model.description}
                                    </p>

                                    {/* Features Table */}
                                    <div className="grid grid-cols-2 gap-3 mb-5">
                                        <div className="flex items-center gap-2">
                                            <FeatureIcon enabled={model.features.voiceDesign} />
                                            <span className={`text-sm ${model.features.voiceDesign ? 'text-text-primary' : 'text-text-tertiary'}`}>
                                                Voice Design
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <FeatureIcon enabled={model.features.customVoice} />
                                            <span className={`text-sm ${model.features.customVoice ? 'text-text-primary' : 'text-text-tertiary'}`}>
                                                Custom Voice
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <FeatureIcon enabled={model.features.voiceClone} />
                                            <span className={`text-sm ${model.features.voiceClone ? 'text-text-primary' : 'text-text-tertiary'}`}>
                                                Voice Clone
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <FeatureIcon enabled={model.features.streaming} />
                                            <span className={`text-sm ${model.features.streaming ? 'text-text-primary' : 'text-text-tertiary'}`}>
                                                Streaming
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <FeatureIcon enabled={model.features.instructionControl} />
                                            <span className={`text-sm ${model.features.instructionControl ? 'text-text-primary' : 'text-text-tertiary'}`}>
                                                Instruction Control
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <Zap className="w-4 h-4 text-yellow-400" />
                                            <span className="text-sm text-text-secondary">{model.memoryRequired}</span>
                                        </div>
                                    </div>

                                    {/* Languages */}
                                    <div className="mb-5">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Globe className="w-4 h-4 text-text-tertiary" />
                                            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                                                Supported Languages
                                            </span>
                                        </div>
                                        <div className="flex flex-wrap gap-1.5">
                                            {model.languages.map((lang) => (
                                                <span
                                                    key={lang}
                                                    className="px-2 py-0.5 rounded text-xs bg-bg-tertiary text-text-secondary"
                                                >
                                                    {lang}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Action hint */}
                                    <div className="pt-4 border-t border-border-default">
                                        <div className="flex items-center justify-between text-xs text-text-tertiary">
                                            <span className="flex items-center gap-1">
                                                <AlertCircle className="w-3.5 h-3.5" />
                                                Update docker-compose.yml QWEN_TTS_MODEL_SIZE to switch
                                            </span>
                                            <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Configuration Guide */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="mt-8 p-6 bg-bg-secondary rounded-2xl border border-border-default"
                    >
                        <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                            <MessageSquare className="w-5 h-5 text-accent-primary" />
                            Configuration Guide
                        </h3>
                        <div className="space-y-4 text-sm text-text-secondary">
                            <p>
                                To change the active model, update your <code className="px-2 py-0.5 bg-bg-tertiary rounded text-accent-primary">docker-compose.yml</code>:
                            </p>
                            <pre className="p-4 bg-bg-tertiary rounded-lg overflow-x-auto text-xs">
                                {`qwen-tts:
  environment:
    - QWEN_TTS_MODEL_SIZE=0.6B        # or 1.7B
    - QWEN_TTS_ENABLE_CUSTOM_VOICE=true
    - QWEN_TTS_ENABLE_VOICE_DESIGN=false  # Only 1.7B
    - QWEN_TTS_ENABLE_VOICE_CLONE=true`}
                            </pre>
                            <p className="text-text-tertiary">
                                After changing, run <code className="px-2 py-0.5 bg-bg-tertiary rounded">docker compose up -d qwen-tts</code> to apply.
                            </p>
                        </div>
                    </motion.div>
                </div>
            </div>
        </Layout>
    );
};

export default ModelsPage;
