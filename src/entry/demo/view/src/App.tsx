import React from 'react';
import { BrowserRouter as Router, Routes, Route, useParams } from 'react-router-dom';
import { WorkspaceSelector } from './pages/WorkspaceSelector';
import { WorkspaceApp } from './pages/WorkspaceApp';

const App: React.FC = () => {
    return (
        <Router>
            <Routes>
                {/* Root - workspace selection */}
                <Route path="/" element={<WorkspaceSelector />} />

                {/* Any path becomes a workspace */}
                <Route path="/:workspaceId" element={<WorkspaceAppWrapper />} />
            </Routes>
        </Router>
    );
};

const WorkspaceAppWrapper: React.FC = () => {
    const { workspaceId } = useParams<{ workspaceId: string }>();

    if (!workspaceId) {
        return <div>Invalid workspace</div>;
    }

    return <WorkspaceApp workspaceId={workspaceId} />;
};

export default App;
