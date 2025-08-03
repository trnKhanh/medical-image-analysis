import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Input, Card, Typography, Space, Alert } from 'antd';
import { FolderOutlined, ArrowRightOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

export const WorkspaceSelector: React.FC = () => {
    const [workspaceId, setWorkspaceId] = useState('');
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    const validateWorkspaceId = (id: string): boolean => {
        return /^[a-zA-Z0-9-]+$/.test(id) && id.length >= 1 && id.length <= 50;
    };

    const handleEnterWorkspace = () => {
        if (!workspaceId.trim()) {
            setError('Please enter a workspace ID');
            return;
        }

        if (!validateWorkspaceId(workspaceId)) {
            setError('Workspace ID can only contain letters, numbers, and hyphens');
            return;
        }

        // Just redirect - no backend call needed
        navigate(`/${workspaceId}`);
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleEnterWorkspace();
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '');
        setWorkspaceId(value);
        if (error) setError(null);
    };

    return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
            <div className="max-w-md w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <Title level={1}>Active Learning Platform</Title>
                    <Text type="secondary" className="text-lg">
                        Enter any workspace name to begin
                    </Text>
                </div>

                {/* Workspace Input */}
                <Card className="shadow-lg">
                    <Space direction="vertical" className="w-full" size="large">
                        <div>
                            <Input
                                size="large"
                                placeholder="my-project"
                                value={workspaceId}
                                onChange={handleInputChange}
                                onKeyPress={handleKeyPress}
                                prefix={<FolderOutlined className="text-gray-400" />}
                                suffix={
                                    workspaceId && validateWorkspaceId(workspaceId) ? (
                                        <ArrowRightOutlined className="text-blue-500 cursor-pointer" onClick={handleEnterWorkspace} />
                                    ) : null
                                }
                                maxLength={50}
                                className="text-center font-mono"
                            />
                            <div className="mt-2 text-center">
                                <Text type="secondary" className="text-sm">
                                    host.com/<span className="font-mono text-blue-600">{workspaceId || 'workspace-name'}</span>
                                </Text>
                            </div>
                        </div>

                        {error && (
                            <Alert
                                message={error}
                                type="error"
                                showIcon
                                closable
                                onClose={() => setError(null)}
                            />
                        )}

                        <Button
                            type="primary"
                            size="large"
                            icon={<ArrowRightOutlined />}
                            onClick={handleEnterWorkspace}
                            disabled={!workspaceId.trim() || !validateWorkspaceId(workspaceId)}
                            className="w-full"
                        >
                            Enter Workspace
                        </Button>

                        {/* Examples */}
                        <div className="mt-6 pt-4 border-t border-gray-200">
                            <Text type="secondary" className="text-sm block mb-2">Examples:</Text>
                            <div className="space-y-1">
                                {['my-project', 'experiment-2024', 'dataset-v2'].map((example) => (
                                    <Button
                                        key={example}
                                        type="link"
                                        size="small"
                                        onClick={() => setWorkspaceId(example)}
                                        className="text-blue-500 p-0 h-auto font-mono"
                                    >
                                        {example}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </Space>
                </Card>

                {/* Info */}
                <div className="text-center mt-6">
                    <Text type="secondary" className="text-sm">
                        Just like dontpad.com - any URL becomes your workspace
                    </Text>
                </div>
            </div>
        </div>
    );
};