import React from 'react';
import { Button, Space } from 'antd';
import { ReloadOutlined, LoadingOutlined, DatabaseOutlined } from '@ant-design/icons';
import { Brain } from 'lucide-react';
import type { SystemStatus } from '../../models';

interface HeaderProps {
    status: SystemStatus;
    onReset: () => void;
    isResetting: boolean;
}

export const Header: React.FC<HeaderProps> = ({ status, onReset, isResetting }) => {
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

                    {/* Right side - Stats and Reset */}
                    <div className="flex items-center space-x-6">
                        {/* Stats */}
                        <Space size="large" className="hidden sm:flex">
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <DatabaseOutlined className="text-blue-500" />
                                <span className="font-medium">Train:</span>
                                <span className="font-mono text-blue-600">{status.train_set_size}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="font-medium">Pool:</span>
                                <span className="font-mono text-green-600">{status.pool_set_size}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="font-medium">Selected:</span>
                                <span className="font-mono text-purple-600">{status.selected_set_size}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <span className="font-medium">Annotated:</span>
                                <span className="font-mono text-orange-600">{status.annotated_set_size}</span>
                            </div>
                        </Space>

                        {/* Mobile Stats - Compact */}
                        <div className="flex sm:hidden items-center space-x-5 text-xs text-gray-600 m-1">
                            <span>Train: <span className="font-mono text-blue-600">{status.train_set_size}</span></span>
                            <span>Pool: <span className="font-mono text-green-600">{status.pool_set_size}</span></span>
                            <span>Select: <span className="font-mono text-purple-600">{status.selected_set_size}</span></span>
                            <span>Annotated: <span className="font-mono text-orange-600">{status.annotated_set_size}</span></span>
                        </div>

                        <Button
                            danger
                            icon={isResetting ? <LoadingOutlined /> : <ReloadOutlined />}
                            loading={isResetting}
                            onClick={onReset}
                            className="flex items-center space-x-5"
                        >
                            <span className="hidden sm:inline ml-1">Reset</span>
                        </Button>
                    </div>
                </div>
            </div>
        </header>
    );
};

