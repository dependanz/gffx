// The dashboard's mesh viewport. A display, not a gffx operation: it claims no conformance.

struct Uniforms {
  mvp : mat4x4<f32>,
  model_view : mat4x4<f32>,
  // x: 1 when per-vertex colour is shown, y: 1 for the wireframe pass, z: picked vertex index or -1
  flags : vec4<f32>,
  base_color : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
  @builtin(vertex_index) index : u32,
};
struct VertexOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) view_normal : vec3<f32>,
  @location(1) color : vec3<f32>,
  @location(2) view_position : vec3<f32>,
};

@vertex
fn vs_main(v : VertexIn) -> VertexOut {
  var out : VertexOut;
  let world = vec4<f32>(v.position, 1.0);
  out.clip = u.mvp * world;
  if (u.flags.y > 0.5) {
    // The wireframe sits a hair in front of the surface so it is not swallowed by depth equality.
    out.clip.z = out.clip.z - 0.0008 * out.clip.w;
  }
  out.view_normal = (u.model_view * vec4<f32>(v.normal, 0.0)).xyz;
  out.view_position = (u.model_view * world).xyz;
  out.color = select(u.base_color.rgb, v.color, u.flags.x > 0.5);
  return out;
}

@fragment
fn fs_main(f : VertexOut) -> @location(0) vec4<f32> {
  if (u.flags.y > 0.5) {
    return vec4<f32>(0.08, 0.08, 0.09, 1.0);
  }
  var n = normalize(f.view_normal);
  // Two-sided: a back face lit as its front so an open mesh reads from both sides.
  if (dot(n, -f.view_position) < 0.0) { n = -n; }
  let light = normalize(vec3<f32>(0.35, 0.55, 1.0));  // a headlight, slightly above and to the right
  let lambert = max(dot(n, light), 0.0);
  let ambient = 0.28;
  let shade = ambient + (1.0 - ambient) * lambert;
  let half = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
  let spec = pow(max(dot(n, half), 0.0), 40.0) * 0.12;
  return vec4<f32>(f.color * shade + vec3<f32>(spec), 1.0);
}

// Instanced camera-facing discs. Points are not fake triangle topology.
struct PointIn {
  @location(0) position : vec3<f32>,
  @location(2) color : vec3<f32>,
  @location(3) radius : f32,
  @builtin(vertex_index) corner : u32,
};
struct PointOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec3<f32>,
};
@vertex fn vs_point(v:PointIn) -> PointOut {
  let corners=array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
  let uv=corners[v.corner];
  let right=vec3(u.model_view[0].x,u.model_view[1].x,u.model_view[2].x);
  let up=vec3(u.model_view[0].y,u.model_view[1].y,u.model_view[2].y);
  var out:PointOut;
  out.clip=u.mvp*vec4(v.position+(right*uv.x+up*uv.y)*v.radius,1.);
  out.uv=uv;out.color=select(u.base_color.rgb,v.color,u.flags.x>0.5);return out;
}
@fragment fn fs_point(v:PointOut) -> @location(0) vec4<f32> {
  if(dot(v.uv,v.uv)>1.){discard;}
  return vec4(v.color,1.);
}
