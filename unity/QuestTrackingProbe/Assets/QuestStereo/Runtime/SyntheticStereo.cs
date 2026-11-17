using UnityEngine;

namespace AbstractStack.QuestStereo
{
    public static class SyntheticStereo
    {
        public static Texture2D Create(bool leftEye, long pairId)
        {
            const int width = 320, height = 240;
            var pixels = new Color32[width * height];
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
            {
                bool grid = x % 20 < 2 || y % 20 < 2;
                pixels[y * width + x] = grid ? new Color32(90, 100, 110, 255) : new Color32(15, 20, 28, 255);
                if (x < 12 || x >= width - 12) pixels[y * width + x] = leftEye ? new Color32(255, 45, 45, 255) : new Color32(40, 100, 255, 255);
                int cx = width / 2 + (leftEye ? 6 : -6), cy = height / 2;
                if ((Mathf.Abs(x - cx) < 3 && Mathf.Abs(y - cy) < 25) || (Mathf.Abs(y - cy) < 3 && Mathf.Abs(x - cx) < 25)) pixels[y * width + x] = Color.white;
                // White moving bar and binary pair counter match in both eyes.
                if (x >= (int)(pairId % 270) + 20 && x < (int)(pairId % 270) + 24 && y < 40) pixels[y * width + x] = Color.yellow;
                if (y > 210 && x > 20 && x < 150 && ((pairId >> ((x - 20) / 8)) & 1) != 0) pixels[y * width + x] = Color.green;
                // Large L/R glyph, deliberately in each texture rather than a shared overlay.
                int gx = x - 25, gy = y - 150;
                bool letter = leftEye ? (gx >= 0 && gx < 6 && gy >= 0 && gy < 35) || (gy >= 0 && gy < 6 && gx >= 0 && gx < 25) :
                    (gx >= 0 && gx < 6 && gy >= 0 && gy < 35) || (gy > 28 && gy < 35 && gx >= 0 && gx < 23) || (gy > 15 && gy < 21 && gx >= 0 && gx < 23) || (gx > 18 && gx < 25 && gy > 15 && gy < 35) || (gy >= 0 && gy < 16 && gx >= 12 - gy / 2 && gx < 19 - gy / 2);
                if (letter) pixels[y * width + x] = Color.white;
            }
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false, false);
            texture.SetPixels32(pixels); texture.Apply(false, true); return texture;
        }
    }
}
