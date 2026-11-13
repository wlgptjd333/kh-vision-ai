// 1일차·2일차 공통: GLB 불러오기 / 재생 / 3D 관절 좌표 추출
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function createViewer(container) {
  // ----------------------------------
  // 1. Three.js 화면 생성
  // ----------------------------------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202530);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(5, 10, 5);
  scene.add(light);

  const grid = new THREE.GridHelper(10, 20, 0x5a6578, 0x353c4a);
  scene.add(grid);

  const controls = new OrbitControls(camera, renderer.domElement);
  const clock = new THREE.Clock();

  const state = {
    name: null, model: null, mixer: null, clip: null, action: null,
    bones: [], playing: true,
  };

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // ----------------------------------
  // 2. 이전 모델 정리 (동작을 바꿀 때 메모리 해제)
  // ----------------------------------
  function unload() {
    if (!state.model) return;
    state.mixer.stopAllAction();
    scene.remove(state.model);
    state.model.traverse(o => {
      o.geometry?.dispose();
      const mats = o.material ? [].concat(o.material) : [];
      for (const m of mats) {
        for (const v of Object.values(m)) if (v?.isTexture) v.dispose();
        m.dispose();
      }
    });
    Object.assign(state, { name: null, model: null, mixer: null, clip: null, action: null, bones: [] });
  }

  // ----------------------------------
  // 3. GLB 불러오기
  // ----------------------------------
  async function load(name) {
    unload();
    const gltf = await new GLTFLoader().loadAsync(`/assets/${encodeURIComponent(name)}.glb`);

    const clip = gltf.animations[0];
    if (!clip) throw new Error("GLB에 애니메이션이 없습니다.");

    const model = gltf.scene;
    const bones = [];
    model.traverse(o => { if (o.isBone) bones.push(o); });
    if (bones.length === 0) throw new Error("GLB에 Bone이 없습니다.");

    scene.add(model);
    const mixer = new THREE.AnimationMixer(model);
    const action = mixer.clipAction(clip);
    action.play();
    mixer.setTime(0);
    model.updateMatrixWorld(true);

    Object.assign(state, { name, model, mixer, clip, action, bones });
    frameCamera();
    return { duration: clip.duration, boneCount: bones.length, clipName: clip.name };
  }

  // 뼈 위치로 범위를 계산해서 카메라를 맞춤
  function frameCamera() {
    const box = new THREE.Box3();
    const p = new THREE.Vector3();
    for (const b of state.bones) box.expandByPoint(b.getWorldPosition(p));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const d = Math.max(size.x, size.y, size.z, 1);
    camera.position.set(center.x + d * 0.6, center.y + d * 0.3, center.z + d * 1.8);
    controls.target.copy(center);
    controls.update();
    grid.position.y = box.min.y;
  }

  // ----------------------------------
  // 4. 재생 제어
  // ----------------------------------
  const tickHandlers = [];

  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    if (state.mixer && state.playing) state.mixer.update(delta);
    controls.update();
    renderer.render(scene, camera);
    for (const fn of tickHandlers) fn(getTime());
  }
  animate();

  function getTime() { return state.action ? state.action.time : 0; }
  function setPlaying(v) { state.playing = v; }
  function restart() { if (state.mixer) { state.mixer.setTime(0); } }

  // ----------------------------------
  // 5. 3D JSON 추출 (노트 3번 6장과 같은 형식)
  // ----------------------------------
  function extract(fps = 30) {
    const { mixer, clip, model, bones, name } = state;
    if (!mixer) throw new Error("먼저 GLB를 불러오세요.");

    const previousTime = getTime();
    const totalFrames = Math.ceil(clip.duration * fps);
    const position = new THREE.Vector3();
    const frames = [];

    try {
      for (let i = 0; i < totalFrames; i++) {
        const time = i / fps;
        mixer.setTime(time);            // 해당 시간의 자세 적용
        model.updateMatrixWorld(true);  // 월드 행렬 갱신 (안 하면 모든 프레임 좌표가 같아짐)

        const joints = bones.map(bone => {
          bone.getWorldPosition(position);
          return {
            name: bone.name,
            x: Number(position.x.toFixed(6)),
            y: Number(position.y.toFixed(6)),
            z: Number(position.z.toFixed(6)),
          };
        });
        frames.push({ frame: i, time_sec: Number(time.toFixed(6)), joints });
      }
    } finally {
      mixer.setTime(previousTime);
      model.updateMatrixWorld(true);
    }

    return {
      source_glb: `${name}.glb`,
      animation: clip.name,
      fps,
      duration_sec: clip.duration,
      coordinate_system: "Three.js world Y-up",
      unit: "meter (glTF standard)",
      bone_count: bones.length,
      frame_count: frames.length,
      frames,
    };
  }

  return {
    load, extract, getTime, setPlaying, restart,
    onTick: fn => tickHandlers.push(fn),
    get duration() { return state.clip ? state.clip.duration : 0; },
    get playing() { return state.playing; },
  };
}
