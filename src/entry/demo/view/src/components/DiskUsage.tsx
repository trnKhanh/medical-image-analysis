import React from 'react';
import {Card, Progress, Typography, Space, Statistic, Button, Spin} from 'antd';
import {FolderOutlined, FileOutlined, WarningOutlined, ReloadOutlined} from '@ant-design/icons';
import type {DiskInfo} from "../models";

const { Text } = Typography;

interface DiskUsageProps {
    diskInfo: DiskInfo | null;
    loading: boolean;
    onRefresh: () => void;
}

export const DiskUsageCard: React.FC<DiskUsageProps> = ({
                                                            diskInfo,
                                                            loading,
                                                            onRefresh
                                                        }) => {
    if (!diskInfo) {
        return (
            <Card
                title="Workspace Disk"
                size="small"
                loading={loading}
                extra={
                    <Button
                        type="text"
                        icon={loading ? <Spin size="small" /> : <ReloadOutlined />}
                        onClick={onRefresh}
                        disabled={loading}
                        title="Refresh disk state"
                    />
                }
            >
                <Text type="secondary">No storage information available</Text>
            </Card>
        );
    }

    const maxSizeMB = Math.round(diskInfo.max_size / (1024 * 1024));
    const usagePercentage = (diskInfo.total_size_mb / maxSizeMB) * 100;
    const isNearLimit = usagePercentage > 80;
    const isAtLimit = usagePercentage > 95;

    const getProgressStatus = () => {
        if (isAtLimit) return 'exception';
        if (isNearLimit) return 'active';
        return 'normal';
    };

    const getProgressColor = () => {
        if (isAtLimit) return '#ff4d4f';
        if (isNearLimit) return '#faad14';
        return '#52c41a';
    };

    const formatBytes = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    };

    return (
        <Card
            title={
                <div className="flex items-center space-x-2">
                    <FolderOutlined />
                    <span>Workspace Storage</span>
                    {isNearLimit && (
                        <WarningOutlined className="text-orange-500" />
                    )}
                </div>
            }
            size="small"
            extra={
                <Button
                    type="text"
                    icon={loading ? <Spin size="small" /> : <ReloadOutlined />}
                    onClick={onRefresh}
                    disabled={loading}
                    title="Refresh disk state"
                />
            }
        >
            <Space direction="vertical" className="w-full" size="middle">
                {/* Progress Bar */}
                <div>
                    <div className="flex justify-between items-center mb-2">
                        <Text strong>Storage Usage</Text>
                        <Text type={isNearLimit ? "warning" : "secondary"}>
                            {diskInfo.total_size_mb.toFixed(1)} MB / {maxSizeMB} MB
                        </Text>
                    </div>
                    <Progress
                        percent={usagePercentage}
                        status={getProgressStatus()}
                        strokeColor={getProgressColor()}
                        showInfo={false}
                    />
                    <div className="flex justify-between mt-1">
                        <Text type="secondary" className="text-xs">
                            {usagePercentage.toFixed(1)}% used
                        </Text>
                        <Text type="secondary" className="text-xs">
                            {(maxSizeMB - diskInfo.total_size_mb).toFixed(1)} MB remaining
                        </Text>
                    </div>
                </div>

                {/* Statistics */}
                <div className="grid grid-cols-2 gap-4">
                    <Statistic
                        title="Total Size"
                        value={formatBytes(diskInfo.total_size)}
                        prefix={<FolderOutlined />}
                        valueStyle={{ fontSize: '14px' }}
                    />
                    <Statistic
                        title="File Count"
                        value={diskInfo.file_count}
                        prefix={<FileOutlined />}
                        valueStyle={{ fontSize: '14px' }}
                    />
                </div>

                {/* Warning Messages */}
                {isAtLimit && (
                    <div className="bg-red-50 border border-red-200 rounded p-2">
                        <Text type="danger" className="text-xs">
                            ⚠️ Storage almost full! Consider deleting unused files.
                        </Text>
                    </div>
                )}
                {isNearLimit && !isAtLimit && (
                    <div className="bg-orange-50 border border-orange-200 rounded p-2">
                        <Text className="text-orange-600 text-xs">
                            ⚡ Storage usage is high ({usagePercentage.toFixed(1)}%)
                        </Text>
                    </div>
                )}
            </Space>
        </Card>
    );
};
