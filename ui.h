// Minimal immediate-mode UI on top of legacy OpenGL.
// Rendered in screen-pixel coords (origin at top-left, y goes down).
// Text uses an embedded font8x8 bitmap; widgets are drawn with GL_QUADS.

#pragma once

namespace flipcpu_ui {

struct Input {
    int  screenW    = 0;
    int  screenH    = 0;
    int  mouseX     = 0;     // screen pixels, y-down
    int  mouseY     = 0;
    bool mouseDown  = false; // current state (LMB)
    bool mousePressed  = false; // edge: pressed this frame
    bool mouseReleased = false; // edge: released this frame
};

// Call once per frame around UI drawing. setProjectionToPixels() switches the
// active GL projection to screen-pixel space; restoreProjection() puts the
// caller's matrices back so the next sim render is unaffected.
void setProjectionToPixels(int width, int height);
void restoreProjection();

void begin(const Input& in);

// Containers
void beginPanel(int x, int y, int w, int h, const char* title);
void endPanel();

// Widgets — all calls advance the panel cursor vertically.
void  text   (const char* fmt, ...);
bool  checkbox(const char* label, bool* value);
bool  slider (const char* label, float* value, float lo, float hi);
bool  button (const char* label);

// True if the mouse is currently inside any UI region (suppress sim input).
bool wantsMouse();

} // namespace flipcpu_ui
