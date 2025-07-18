import React from 'react';
import { Button, Space } from 'antd';
import { ReloadOutlined, LoadingOutlined, DatabaseOutlined, SyncOutlined } from '@ant-design/icons';
import { Brain } from 'lucide-react';
import type { ActiveLearningState } from '../../models';

interface HeaderProps {
    status: ActiveLearningState;
    onReset: () => void;
    isResetting: boolean;
    onSync: () => void;
    isSyncing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
                                                  status,
                                                  onReset,
                                                  isResetting,
                                                  onSync,
                                                  isSyncing
                                              }) => {
    return (
        <header className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 py-6">
                <div className="flex items-center justify-between">
                    {/* Left side - Title */}
                    <div className="flex items-center space-x-3">
                        <Brain className="h-8 w-8 text-blue-600" />
                        <h1 className="text-2xl font-bold text-gray-900 m-0">
                            Active Learning Annotation
                        </h1>
                    </div>

                    {/* Right side - Stats and Actions */}
                    <div className="flex items-center space-x-6">
                        {/* Stats */}
                        <Space size="large" className="hidden sm:flex">
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <DatabaseOutlined className="text-blue-500" />
                                <span className="font-medium">Train:</span>
                                <span className="font-mono text-blue-600">{status.train_count}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="font-medium">Pool:</span>
                                <span className="font-mono text-green-600">{status.pool_count}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="font-medium">Annotated:</span>
                                <span className="font-mono text-orange-600">{status.annotated_count}</span>
                            </div>
                        </Space>

                        {/* Mobile Stats */}
                        <div className="flex sm:hidden items-center space-x-5 text-xs text-gray-600 m-1">
                            <span>Train: <span className="font-mono text-blue-600">{status.train_count}</span></span>
                            <span>Pool: <span className="font-mono text-green-600">{status.pool_count}</span></span>
                            <span>Annotated: <span className="font-mono text-orange-600">{status.annotated_count}</span></span>
                        </div>

                        {/* Sync Button */}
                        <Button
                            icon={isSyncing ? <LoadingOutlined /> : <SyncOutlined />}
                            loading={isSyncing}
                            onClick={onSync}
                            className="flex items-center"
                        >
                            <span className="hidden sm:inline ml-1">Sync</span>
                        </Button>

                        {/* Reset Button */}
                        <Button
                            danger
                            icon={isResetting ? <LoadingOutlined /> : <ReloadOutlined />}
                            loading={isResetting}
                            onClick={onReset}
                            className="flex items-center"
                        >
                            <span className="hidden sm:inline ml-1">Reset</span>
                        </Button>
                    </div>
                </div>
            </div>
        </header>
    );
};
