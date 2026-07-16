// API Error Handling
// Standardized error responses

export interface ApiError {
  error: string
  message: string
  details?: unknown
}

/**
 * Create a standardized API error response.
 */
export function createApiError(
  error: string,
  message: string,
  details?: unknown
): ApiError {
  const result: ApiError = { error, message }
  if (details !== undefined) {
    result.details = details
  }
  return result
}

/**
 * Common API errors
 */
export const ApiErrors = {
  VALIDATION_ERROR: (details: unknown) =>
    createApiError('validation_error', 'Request validation failed', details),

  MODEL_NOT_FOUND: (modelName: string) =>
    createApiError('model_not_found', `Model "${modelName}" not found in catalog`),

  GPU_NOT_FOUND: (gpuType: string) =>
    createApiError('gpu_not_found', `GPU "${gpuType}" not found in catalog`),

  INTERNAL_ERROR: (message: string) =>
    createApiError('internal_error', message),

  INVALID_REQUEST: (message: string) =>
    createApiError('invalid_request', message),

  GPU_SIZER_NOT_CONFIGURED: () =>
    createApiError('gpu_sizer_not_configured', 'GPU sizing service is not configured'),

  GPU_SIZER_AUTH_FAILED: () =>
    createApiError('gpu_sizer_auth_failed', 'Authentication with GPU sizing service failed'),

  GPU_SIZER_UNAVAILABLE: (detail?: string) =>
    createApiError('gpu_sizer_unavailable', detail ?? 'GPU sizing service is unreachable'),

  GPU_SIZER_TIMEOUT: () =>
    createApiError('gpu_sizer_timeout', 'The GPU sizing engine took longer than expected (90s timeout)'),

  GPU_SIZER_INVALID_RESPONSE: (detail?: string) =>
    createApiError('gpu_sizer_invalid_response', detail ?? 'GPU sizing service returned an invalid response'),
}
