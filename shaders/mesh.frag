#version 330 core

in vec3 frag_normal;
in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient;

out vec4 out_color;

void main() {
    vec3 norm = normalize(frag_normal);
    float diff = max(dot(norm, -light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    vec3 result = (ambient + diffuse) * frag_color;
    out_color = vec4(result, 1.0);
}
