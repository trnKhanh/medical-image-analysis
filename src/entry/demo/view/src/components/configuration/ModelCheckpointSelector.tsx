import React from 'react';
import { Select, Button, Space, Typography, Spin } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { formatFileSize } from '../../commons/utils';
import type { ModelCheckpoint } from '../../models';

const { Option } = Select;
const { Text } = Typography;

interface ModelCheckpointSelectorProps {
    value: string;
    checkpoints: ModelCheckpoint[];
    loading: boolean;
    onChange: (value: string) => void;
    onRefresh: () => void;
}

export const ModelCheckpointSelector: React.FC<ModelCheckpointSelectorProps> = ({
                                                                                    value,
                                                                                    checkpoints,
                                                                                    loading,
                                                                                    onChange,
                                                                                    onRefresh,
                                                                                }) => {
    const selectedCheckpoint = checkpoints.find((cp) => cp.name === value);

    return (
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <Space.Compact style={{ width: '100%' }}>
                <Select
                    placeholder="Select a checkpoint…"
                    value={value || undefined}
                    onChange={onChange}
                    loading={loading}
                    disabled={loading}
                    style={{ flex: 1 }}
                    showSearch
                    optionFilterProp="children"
                    allowClear
                >
                    {checkpoints.map((checkpoint) => (
                        <Option key={checkpoint.name} value={checkpoint.name}>
                            {checkpoint.name} ({formatFileSize(checkpoint.size)})
                        </Option>
                    ))}
                </Select>

                <Button
                    icon={loading ? <Spin size="small" /> : <ReloadOutlined />}
                    onClick={onRefresh}
                    disabled={loading}
                    title="Refresh checkpoint list"
                />
            </Space.Compact>

            {selectedCheckpoint && (
                <div
                    style={{
                        marginTop: 8,
                        padding: 12,
                        background: '#fafafa',
                        borderRadius: 4,
                        fontSize: 12,
                    }}
                >
                    <div>
                        <Text strong>File:</Text> {selectedCheckpoint.name}
                    </div>
                    <div>
                        <Text strong>Size:</Text> {formatFileSize(selectedCheckpoint.size)}
                    </div>
                </div>
            )}
        </Space>
    );
};
