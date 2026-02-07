//! Milvus REST auth header helper.

pub struct MilvusRestAuthInput<'a> {
    pub token: Option<&'a str>,
    pub username: Option<&'a str>,
    pub password: Option<&'a str>,
}

pub fn build_rest_auth_header(input: &MilvusRestAuthInput<'_>) -> Option<String> {
    let token = input.token.map(str::trim).filter(|token| !token.is_empty());
    if let Some(token) = token {
        return Some(format!("Bearer {token}"));
    }

    let username = input
        .username
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let password = input.password.filter(|value| !value.is_empty());

    match (username, password) {
        (Some(username), Some(password)) => Some(format!("Bearer {username}:{password}")),
        _ => None,
    }
}
