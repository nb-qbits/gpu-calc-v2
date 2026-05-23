module.exports = {
  '*.{ts,tsx}': [
    'eslint --fix',
    () => 'tsc --noEmit',
  ],
}
