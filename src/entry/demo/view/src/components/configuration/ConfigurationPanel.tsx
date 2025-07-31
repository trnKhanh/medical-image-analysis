import { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { SettingOutlined, SaveOutlined } from '@ant-design/icons';
import { Card, Form, InputNumber, Select, Row, Col, Button, Space } from 'antd';
import type { Config, ModelCheckpoint } from '../../models';
import { ModelCheckpointSelector } from './ModelCheckpointSelector';

const { Option } = Select;

interface ConfigurationPanelProps {
    config: Config;
    checkpoints: ModelCheckpoint[];
    loadingCheckpoints: boolean;
    onRefreshCheckpoints: () => void;
    onUpdateConfig?: (config: Config) => Promise<void>;
}

export interface ConfigurationPanelRef {
    getCurrentConfig: () => Config;
}

export const ConfigurationPanel = forwardRef<ConfigurationPanelRef, ConfigurationPanelProps>(({
                                                                                                  config,
                                                                                                  checkpoints,
                                                                                                  loadingCheckpoints,
                                                                                                  onRefreshCheckpoints,
                                                                                                  onUpdateConfig
                                                                                              }, ref) => {
    const [form] = Form.useForm();
    const [localConfig, setLocalConfig] = useState<Config>(config);
    const [updating, setUpdating] = useState(false);
    const [hasChanges, setHasChanges] = useState(false);

    useEffect(() => {
        setLocalConfig(config);
        setHasChanges(false);
        console.log("Full config object:", config);
        console.log("Config keys:", Object.keys(config));
        console.log("MODEL", config.model_ckpt);
        console.log("DEVICE", config.device);

        form.setFieldsValue({
            model_ckpt: config.model_ckpt,
            device: config.device,
            batch_size: config.batch_size,
            loaded_feature_weight: config.loaded_feature_weight,
            budget: config.budget,
            sharp_factor: config.sharp_factor,
        });
    }, [config, form]);

    useEffect(() => {
        const configChanged = JSON.stringify(localConfig) !== JSON.stringify(config);
        setHasChanges(configChanged);
    }, [localConfig, config]);

    useImperativeHandle(ref, () => ({
        getCurrentConfig: () => localConfig
    }), [localConfig]);

    const handleLocalConfigChange = (partialConfig: Partial<Config>) => {
        const newConfig = { ...localConfig, ...partialConfig };
        setLocalConfig(newConfig);

        form.setFieldsValue(partialConfig);
    };

    const handleUpdateConfig = async () => {
        if (!onUpdateConfig) {
            console.error('Update config not provided');
            return;
        }

        try {
            await form.validateFields();

            setUpdating(true);
            await onUpdateConfig(localConfig);
            setUpdating(false);
        } catch (error) {
            console.error('Form validation failed:', error);
            setUpdating(false);
        }
    };

    const handleReset = () => {
        setLocalConfig(config);
        setHasChanges(false);

        form.setFieldsValue({
            model_ckpt: config.model_ckpt,
            device: config.device,
            batch_size: config.batch_size,
            loaded_feature_weight: config.loaded_feature_weight,
            budget: config.budget,
            sharp_factor: config.sharp_factor,
        });
    };

    const validateConfig = () => {
        const requiredFields = ['model_ckpt', 'device', 'batch_size', 'loaded_feature_weight', 'budget', 'sharp_factor'];
        const missingFields = requiredFields.filter(field => {
            const value = localConfig[field as keyof Config];
            return value === undefined || value === null || value === '';
        });

        return missingFields.length === 0;
    };

    const isFormValid = validateConfig();

    return (
        <Card
            title={
                <span>
                    <SettingOutlined style={{ marginRight: 8 }} />
                    Configuration
                </span>
            }
            extra={
                <Space>
                    {hasChanges && (
                        <Button
                            size="small"
                            onClick={handleReset}
                            disabled={updating}
                        >
                            Reset
                        </Button>
                    )}
                    <Button
                        type="primary"
                        icon={<SaveOutlined />}
                        loading={updating}
                        disabled={!hasChanges || !isFormValid}
                        onClick={handleUpdateConfig}
                        size="small"
                    >
                        {updating ? 'Updating...' : 'Update Config'}
                    </Button>
                </Space>
            }
        >
            <Form
                form={form}
                layout="vertical"
                initialValues={{
                    model_ckpt: config.model_ckpt,
                    device: config.device,
                    batch_size: config.batch_size,
                    loaded_feature_weight: config.loaded_feature_weight,
                    budget: config.budget,
                    sharp_factor: config.sharp_factor,
                }}
            >
                <Form.Item
                    label="Model Checkpoint"
                    name="model_ckpt"
                    rules={[{ required: true, message: 'Please select a model checkpoint' }]}
                >
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
                        <Form.Item
                            label="Device"
                            name="device"
                            rules={[{ required: true, message: 'Please select a device' }]}
                        >
                            <Select
                                value={localConfig.device}
                                onChange={(value) => handleLocalConfigChange({ device: value })}
                                disabled={updating}
                                placeholder="Select device"
                            >
                                <Option value="cpu">🔵 CPU</Option>
                                <Option value="cuda">🟢 CUDA</Option>
                            </Select>
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item
                            label="Batch Size"
                            name="batch_size"
                            rules={[{ required: true, message: 'Please select batch size' }]}
                        >
                            <Select
                                value={localConfig.batch_size}
                                onChange={(value) => handleLocalConfigChange({ batch_size: value })}
                                disabled={updating}
                                placeholder="Select batch size"
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

                <Form.Item
                    label="Foundation Model Weight"
                    name="loaded_feature_weight"
                    rules={[
                        { required: true, message: 'Please enter foundation model weight' },
                        { type: 'number', min: 0, message: 'Value must be greater than or equal to 0' }
                    ]}
                >
                    <InputNumber
                        value={localConfig.loaded_feature_weight}
                        step={0.1}
                        min={0}
                        onChange={(value) =>
                            handleLocalConfigChange({ loaded_feature_weight: value ?? 0 })
                        }
                        style={{ width: '100%' }}
                        disabled={updating}
                        placeholder="Enter foundation model weight"
                    />
                </Form.Item>

                <Row gutter={12}>
                    <Col span={12}>
                        <Form.Item
                            label="Budget"
                            name="budget"
                            rules={[
                                { required: true, message: 'Please enter budget' },
                                { type: 'number', min: 0, message: 'Budget must be greater than or equal to 0' }
                            ]}
                        >
                            <InputNumber
                                value={localConfig.budget}
                                onChange={(value) => handleLocalConfigChange({ budget: value ?? 0 })}
                                style={{ width: '100%' }}
                                min={0}
                                disabled={updating}
                                placeholder="Enter budget"
                            />
                        </Form.Item>
                    </Col>
                    <Col span={12}>
                        <Form.Item
                            label="Sharp Factor"
                            name="sharp_factor"
                            rules={[
                                { required: true, message: 'Please enter sharp factor' },
                                { type: 'number', min: 0, message: 'Sharp factor must be greater than or equal to 0' }
                            ]}
                        >
                            <InputNumber
                                value={localConfig.sharp_factor}
                                step={0.1}
                                min={0}
                                onChange={(value) => handleLocalConfigChange({ sharp_factor: value ?? 0 })}
                                style={{ width: '100%' }}
                                disabled={updating}
                                placeholder="Enter sharp factor"
                            />
                        </Form.Item>
                    </Col>
                </Row>

                {!isFormValid && (
                    <div style={{
                        padding: '8px 12px',
                        backgroundColor: '#fff2f0',
                        border: '1px solid #ffccc7',
                        borderRadius: '6px',
                        fontSize: '12px',
                        color: '#cf1322',
                        marginTop: '16px'
                    }}>
                        ❌ Please fill in all required fields
                    </div>
                )}

                {hasChanges && isFormValid && (
                    <div style={{
                        padding: '8px 12px',
                        backgroundColor: '#fff7e6',
                        border: '1px solid #ffd591',
                        borderRadius: '6px',
                        fontSize: '12px',
                        color: '#d46b08',
                        marginTop: '16px'
                    }}>
                        ⚠️ You have unsaved changes
                    </div>
                )}
            </Form>
        </Card>
    );
});