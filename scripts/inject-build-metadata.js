#!/usr/bin/env node

// Inject build metadata into environment variables
// Run this in package.json build script before next build

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Get git commit hash
let gitCommit = 'unknown';
try {
  gitCommit = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
} catch (e) {
  console.warn('⚠️  Could not get git commit hash');
}

// Get build timestamp
const buildTime = new Date().toISOString();

// Write to .env.local for Next.js to pick up
const envPath = path.join(__dirname, '..', '.env.local');
const envContent = `
# Auto-generated build metadata - DO NOT EDIT
NEXT_PUBLIC_BUILD_TIME=${buildTime}
NEXT_PUBLIC_GIT_COMMIT=${gitCommit}
`;

fs.appendFileSync(envPath, envContent);

console.log('✅ Build metadata injected:');
console.log(`   Build time: ${buildTime}`);
console.log(`   Git commit: ${gitCommit.substring(0, 7)}`);
