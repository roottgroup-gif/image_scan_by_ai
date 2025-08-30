#!/usr/bin/env node
/**
 * Bridge script to run Python Flask application from Node.js workflow
 * This allows the existing npm workflow to execute the Python Flask app
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🌉 Starting Node.js -> Python Flask Bridge');
console.log('📱 AI Image Detection Web Application');

// Start the Python Flask application
const pythonProcess = spawn('python', ['main.py'], {
    cwd: __dirname,
    stdio: 'inherit'
});

pythonProcess.on('close', (code) => {
    console.log(`Python Flask app exited with code ${code}`);
    process.exit(code);
});

pythonProcess.on('error', (err) => {
    console.error('Failed to start Python Flask app:', err);
    process.exit(1);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('🛑 Shutting down gracefully...');
    pythonProcess.kill('SIGINT');
});

process.on('SIGTERM', () => {
    console.log('🛑 Terminating...');
    pythonProcess.kill('SIGTERM');
});