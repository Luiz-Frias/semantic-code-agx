# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in semantic-code-agx, please report it responsibly by emailing **security@example.com** rather than using the public issue tracker.

### Reporting Guidelines

1. **Email subject**: Include `[SECURITY]` in the subject line
2. **Vulnerability details**: Describe the vulnerability with sufficient detail for reproduction
3. **Affected versions**: Specify which versions are affected
4. **Suggested fix**: If you have a proposed patch, please include it (but do not publish it publicly)
5. **Your contact information**: Provide email or other contact method for follow-up

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Assessment**: We will assess the severity and scope of the vulnerability
- **Fix development**: We will work on a patch for confirmed vulnerabilities
- **Disclosure**: We will coordinate with you on a responsible disclosure timeline

For critical vulnerabilities, we aim to release a fix within 30 days. For non-critical issues, we will include the fix in the next regular release.

## Security Practices

### Code Review

- All contributions undergo peer review before merging
- Security-sensitive code receives additional scrutiny
- We use automated security scanning tools

### Dependency Management

- Dependencies are regularly audited using `cargo audit`
- We keep dependencies up-to-date with security patches
- Critical dependencies are pinned to specific versions

### Testing

- All code includes unit and integration tests
- Security-critical paths have extensive test coverage
- We use fuzzing for input validation

### Data Handling

- Sensitive data (API keys, tokens) is never logged
- We implement secret redaction in error messages
- Configuration files with credentials should not be committed to git

## Security Advisories

We publish security advisories for vulnerabilities that affect released versions. These are available via:

- GitHub Security Advisories
- Our CHANGELOG.md file
- Direct notification to maintainers

## Third-Party Dependencies

We maintain awareness of security issues in our dependencies through:

- Regular `cargo audit` scans
- GitHub Dependabot alerts
- Manual security audits of critical dependencies

## Compliance

This project follows security best practices recommended by:

- OWASP (Open Web Application Security Project)
- Rust Security Guidelines
- General software supply chain security principles
