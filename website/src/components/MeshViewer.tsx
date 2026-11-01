import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { asset, type Media } from '../data/project';

export default function MeshViewer({ item }: { item: Media }) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ reset: () => void; wireframe: (value: boolean) => void } | null>(null);
  const [status, setStatus] = useState('Loading the research mesh…');
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!host.current || !item.src) return;
    const element = host.current;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | undefined;
    let controls: OrbitControls | undefined;
    let observer: ResizeObserver | undefined;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#101010');
    function disposeScene() {
      scene.traverse(obj => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          materials.forEach(material => { Object.values(material).forEach(value => { if (value instanceof THREE.Texture) value.dispose(); }); material.dispose(); });
        }
      });
    }
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.domElement.tabIndex = 0;
      renderer.domElement.setAttribute('aria-label', 'Interactive object mesh. Drag to orbit, scroll to zoom. Arrow keys orbit, plus and minus zoom.');
      element.appendChild(renderer.domElement);
      const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 1000);
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enablePan = false;
      scene.add(new THREE.HemisphereLight(0xffffff, 0x454545, 2.6));
      const light = new THREE.DirectionalLight(0xffffff, 3);
      light.position.set(3, 5, 4); scene.add(light);
      const render = () => { if (!disposed) renderer?.render(scene, camera); };
      controls.addEventListener('change', render);
      observer = new ResizeObserver(() => { const width = element.clientWidth; const height = element.clientHeight; renderer?.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); render(); });
      observer.observe(element);
      new GLTFLoader().load(asset(item.src), gltf => {
        if (disposed) { scene.add(gltf.scene); disposeScene(); return; }
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        gltf.scene.position.sub(center);
        scene.add(gltf.scene);
        camera.near = size / 1000; camera.far = size * 100; camera.updateProjectionMatrix();
        const reset = () => { camera.position.set(size * 1.1, size * 0.6, size * 1.4); controls!.target.set(0, 0, 0); controls!.update(); render(); };
        api.current = { reset, wireframe: value => { gltf.scene.traverse(obj => { if (obj instanceof THREE.Mesh) { const materials = Array.isArray(obj.material) ? obj.material : [obj.material]; materials.forEach(m => { if ('wireframe' in m) m.wireframe = value; }); } }); render(); } };
        reset();
        setStatus('Mesh loaded. Drag to orbit · scroll to zoom.');
        renderer!.domElement.onkeydown = event => {
          const offset = camera.position.clone().sub(controls!.target);
          const spherical = new THREE.Spherical().setFromVector3(offset);
          if (event.key === 'ArrowLeft') spherical.theta -= 0.12;
          else if (event.key === 'ArrowRight') spherical.theta += 0.12;
          else if (event.key === 'ArrowUp') spherical.phi = Math.max(0.05, spherical.phi - 0.12);
          else if (event.key === 'ArrowDown') spherical.phi = Math.min(Math.PI - 0.05, spherical.phi + 0.12);
          else if (event.key === '+' || event.key === '=') spherical.radius *= 0.9;
          else if (event.key === '-') spherical.radius *= 1.1;
          else return;
          event.preventDefault(); camera.position.copy(new THREE.Vector3().setFromSpherical(spherical)); controls!.update(); render();
        };
      }, undefined, () => { if (!disposed) { setError(true); setStatus('The interactive mesh could not load. Static render shown below.'); } });
    } catch { setError(true); setStatus('WebGL unavailable. Static render shown below.'); }
    return () => { disposed = true; api.current = null; observer?.disconnect(); controls?.dispose(); disposeScene(); renderer?.dispose(); renderer?.domElement.remove(); };
  }, [item]);
  return <div className="mesh-viewer"><div ref={host} className={`mesh-canvas${error ? ' hidden' : ''}`} />{error && item.poster && <img src={asset(item.poster)} alt={item.alt} />}<div className="mesh-toolbar"><button className="quiet-button" disabled={error} onClick={() => api.current?.reset()}>Reset camera</button><label><input disabled={error} type="checkbox" onChange={e => api.current?.wireframe(e.target.checked)} /> Wireframe</label><a href={asset(item.src!)} download>Download GLB ↓</a></div><p className="fine" role="status">{status} Display backup · units unvalidated.</p></div>;
}
