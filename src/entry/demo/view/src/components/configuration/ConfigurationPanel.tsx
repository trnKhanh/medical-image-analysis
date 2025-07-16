import { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { SettingOutlined } from '@ant-design/icons';
import { Card, Form, InputNumber, Select, Row, Col } from 'antd';
import type { Config, ModelCheckpoint } from '../../models';
import { ModelCheckpointSelector } from './ModelCheckpointSelector';

const { Option } = Select;

interface ConfigurationPanelProps {
    config: Config;
    checkpoints: ModelCheckpoint[];
    loadingCheckpoints: boolean;
    onRefreshCheckpoints: () => void;
}

export interface ConfigurationPanelRef {
    getCurrentConfig: () => Config;
}

export const ConfigurationPanel = forwardRef<ConfigurationPanelRef, ConfigurationPanelProps>(({
                                                                                                  config,
                                                                                                  checkpoints,
                                                                                                  loadingCheckpoints,
                                                                                                  onRefreshCheckpoints
                                                                                              }, ref) => {
    const [localConfig, setLocalConfig] = useState<Config>(config);
    useEffect(() => {
        setLocalConfig(config);
    }, [config]);

    useImperativeHandle(ref, () => ({
        getCurrentConfig: () => localConfig
    }), [localConfig]);

    const handleLocalConfigChange = (partialConfig: Partial<Config>) => {
        setLocalConfig(prev => ({ ...prev, ...partialConfig }));
    };

    return (
        <Card
            title={
                <span>
                    <SettingOutlined style={{ marginRight: 8 }} />
                    Configuration
                </span>
            }
        >
            <Form layout="vertical">
                <Form.Item label="Model Checkpoint">
                    <ModelCheckpointSelector
                        value={localConfig.model_ckpt}
                        checkpoints={checkpoints}
                        loading={loadingCheckpoints}
                        onChange={(value) => handleLocalConfigChange({ model_ckpt: value })}
                        onRefresh={onRefreshCheckpoints}
                    />
                </Form.Item>

                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item label="Device">
                            <Select
                                value={localConfig.device}
                                onChange={(value) => handleLocalConfigChange({ device: value })}
                            >
                                <Option value="cpu">🔵 CPU</Option>
                                <Option value="cuda">🟢 CUDA</Option>
                            </Select>
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item label="Batch Size">
                            <Select
                                value={localConfig.batch_size}
                                onChange={(value) => handleLocalConfigChange({ batch_size: value })}
                            >
                                {[4, 8, 16, 32, 64].map((size) => (
                                    <Option key={size} value={size}>
                                        {size}
                                    </Option>
                                ))}
                            </Select>
                        </Form.Item>
                    </Col>
                </Row>

                <Form.Item label="Foundation Model Weight">
                    <InputNumber
                        value={localConfig.loaded_feature_weight}
                        step={0.1}
                        min={0}
                        onChange={(value) =>
                            handleLocalConfigChange({ loaded_feature_weight: value ?? 0 })
                        }
                        style={{ width: '100%' }}
                    />
                </Form.Item>

                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item label="Budget">
                            <InputNumber
                                value={localConfig.budget}
                                onChange={(value) => handleLocalConfigChange({ budget: value ?? 0 })}
                                style={{ width: '100%' }}
                                min={0}
                            />
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item label="Sharp Factor">
                            <InputNumber
                                value={localConfig.sharp_factor}
                                step={0.1}
                                min={0}
                                onChange={(value) => handleLocalConfigChange({ sharp_factor: value ?? 0 })}
                                style={{ width: '100%' }}
                            />
                        </Form.Item>
                    </Col>
                </Row>
            </Form>
        </Card>
    );
});