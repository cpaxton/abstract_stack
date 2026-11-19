// Access-token handling for the teleop host (see docs/web_teleop/README.md).

const TOKEN_KEY = "teleop-token";

export function storeToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage may be unavailable (private mode); the token then lasts for this page only.
  }
}

/**
 * Take `?token=` from the address (then strip it so it is not shared or bookmarked),
 * else reuse the stored one: home-screen launches open "/" without a token.
 */
export function takeToken(): string | null {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    storeToken(fromUrl);
    url.searchParams.delete("token");
    window.history.replaceState(null, "", url.toString());
    return fromUrl;
  }
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}
