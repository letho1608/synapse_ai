// 3D view of the cluster: nodes on a ring in pipeline order, links between
// consecutive stages, particles flowing along them (faster while requests run),
// a ring per node showing its share of the layers, HTML labels projected on top.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ClusterResources, DistributedStatus } from './types';

export interface NodeView {
  id: string;
  name: string;
  slot: number;
  ramGb: number;
  share: number;          // 0..1 of the model's layers
  layers: string;         // "0–15"
  cpu: number | null;
  self: boolean;
  master: boolean;
  standby: boolean;
}

function cssColor(name: string, fallback: string): THREE.Color {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  try { return new THREE.Color(v || fallback); } catch { return new THREE.Color(fallback); }
}

const SLOT_VARS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8'];

export function nodeViews(c: ClusterResources | null, d: DistributedStatus | null, slot: (id: string) => number): NodeView[] {
  if (!c) return [];
  return c.nodes
    .slice()
    .sort((a, b) => (a.layer_start ?? 1e9) - (b.layer_start ?? 1e9))
    .map(n => ({
      id: n.node_id,
      name: n.name,
      slot: slot(n.node_id),
      ramGb: n.ram_total_gb ?? n.ram_gb ?? 1,
      share: (n.share_pct ?? 0) / 100,
      layers: n.layer_start === null ? '—' : `${n.layer_start}–${n.layer_end}`,
      cpu: n.cpu_pct,
      self: n.is_self,
      master: n.node_id === d?.master_id,
      standby: !!d?.standby_nodes?.includes(n.node_id),
    }));
}

interface NodeObj {
  view: NodeView;
  group: THREE.Group;
  core: THREE.Mesh;
  halo: THREE.Mesh;
  ring: THREE.Mesh;
  label: HTMLDivElement;
}

export class Cluster3D {
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  private controls: OrbitControls | null = null;
  private nodes: NodeObj[] = [];
  private links: THREE.Object3D[] = [];
  private curves: THREE.CatmullRomCurve3[] = [];
  private particles: Array<{ mesh: THREE.Mesh; curve: number; t: number }> = [];
  private labels: HTMLDivElement;
  private raf = 0;
  private busy = false;
  private key = '';
  private ro: ResizeObserver;
  private clock = new THREE.Clock();
  private reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  readonly ok: boolean;

  constructor(private host: HTMLElement) {
    host.classList.add('viz3d');
    host.innerHTML = '';
    this.labels = document.createElement('div');
    this.labels.className = 'viz3d-labels';
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      this.renderer = null;
    }
    this.ok = !!this.renderer;
    if (!this.renderer) {
      host.innerHTML = '<div class="muted center">Trình duyệt không hỗ trợ WebGL, không vẽ được sơ đồ 3D.</div>';
      this.ro = new ResizeObserver(() => {});
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    host.append(this.renderer.domElement, this.labels);

    this.camera.position.set(0, 7, 13);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = !this.reduced;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 30;
    this.controls.maxPolarAngle = Math.PI * 0.48;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.PointLight(0x9fd8ff, 60, 60);
    key.position.set(0, 10, 6);
    this.scene.add(key);
    const grid = new THREE.GridHelper(40, 40, 0x1d3550, 0x14243a);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    grid.position.y = -1.6;
    this.scene.add(grid);
    this.scene.fog = new THREE.Fog(0x0a0e1a, 18, 42);

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    this.loop();
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls?.dispose();
    this.clear();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }

  private resize() {
    if (!this.renderer) return;
    const w = Math.max(200, this.host.clientWidth);
    const h = Math.max(200, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private clear() {
    const dispose = (o: THREE.Object3D) => o.traverse(c => {
      const m = c as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach(x => x.dispose()); else mat?.dispose();
    });
    for (const n of this.nodes) { this.scene.remove(n.group); dispose(n.group); n.label.remove(); }
    for (const l of this.links) { this.scene.remove(l); dispose(l); }
    for (const p of this.particles) { this.scene.remove(p.mesh); dispose(p.mesh); }
    this.nodes = []; this.links = []; this.curves = []; this.particles = [];
  }

  update(views: NodeView[], busy: boolean) {
    if (!this.renderer) return;
    this.busy = busy;
    const key = views.map(v => `${v.id}:${v.layers}:${v.share.toFixed(3)}`).join('|');
    if (key !== this.key) { this.key = key; this.rebuild(views); }
    for (const n of this.nodes) {
      const v = views.find(x => x.id === n.view.id);
      if (v) n.view = v;
      n.label.innerHTML = this.labelHtml(n.view);
    }
  }

  private labelHtml(v: NodeView): string {
    const tags = [v.master ? 'master' : '', v.self ? 'máy này' : '', v.standby ? 'dự phòng' : ''].filter(Boolean);
    const esc = (s: string) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
    return `<div class="viz3d-name"><i style="background:var(${SLOT_VARS[Math.min(v.slot, 7)]})"></i>${esc(v.name)}</div>
      <div class="viz3d-meta">layer ${esc(v.layers)} · CPU ${v.cpu === null ? '—' : Math.round(v.cpu) + '%'}</div>
      ${tags.length ? `<div class="viz3d-tags">${tags.map(t => `<span>${t}</span>`).join('')}</div>` : ''}`;
  }

  private rebuild(views: NodeView[]) {
    this.clear();
    const n = views.length;
    const R = n <= 1 ? 0 : Math.max(3.2, n * 1.15);
    const maxRam = Math.max(...views.map(v => v.ramGb), 1);
    const pos: THREE.Vector3[] = [];
    views.forEach((v, i) => {
      // start on the left so the first stages read left to right from the default camera
      const a = (i / Math.max(1, n)) * Math.PI * 2 + Math.PI;
      const p = new THREE.Vector3(Math.cos(a) * R, 0, Math.sin(a) * R);
      pos.push(p);
      const color = cssColor(SLOT_VARS[Math.min(v.slot, 7)], '#3987e5');
      const size = 0.55 + 0.45 * Math.sqrt(v.ramGb / maxRam);
      const group = new THREE.Group();
      group.position.copy(p);
      const core = new THREE.Mesh(
        new THREE.SphereGeometry(size, 40, 28),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.55, roughness: 0.35, metalness: 0.2 }),
      );
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(size * 1.55, 32, 20),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, depthWrite: false }),
      );
      // ring arc = this node's share of the layers
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(size * 1.35, 0.045, 10, 96, Math.max(0.05, v.share) * Math.PI * 2),
        new THREE.MeshBasicMaterial({ color }),
      );
      ring.rotation.x = Math.PI / 2;
      const track = new THREE.Mesh(
        new THREE.TorusGeometry(size * 1.35, 0.02, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x2a3b55, transparent: true, opacity: 0.7 }),
      );
      track.rotation.x = Math.PI / 2;
      // pillar to the floor anchors the node in space
      const pillar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.02, 0.02, 1.6, 6),
        new THREE.MeshBasicMaterial({ color: 0x2a3b55 }),
      );
      pillar.position.y = -0.8;
      group.add(core, halo, track, ring, pillar);
      this.scene.add(group);
      const label = document.createElement('div');
      label.className = 'viz3d-label';
      this.labels.appendChild(label);
      this.nodes.push({ view: v, group, core, halo, ring, label });
    });

    // pipeline links: stage i -> i+1, and the last stage back to the first
    // (the sampled token returns to the first stage for the next step)
    if (n >= 2) {
      for (let i = 0; i < n; i++) {
        const a = pos[i];
        const b = pos[(i + 1) % n];
        const isReturn = i === n - 1;
        const mid = a.clone().add(b).multiplyScalar(0.5);
        mid.y += isReturn ? -0.6 : 1.4;
        // two nodes: forward and return paths would overlap, so bow them apart
        if (n === 2) mid.z += isReturn ? 1.6 : -1.6;
        const curve = new THREE.CatmullRomCurve3([a.clone(), mid, b.clone()]);
        this.curves.push(curve);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, 48, isReturn ? 0.02 : 0.035, 8, false),
          new THREE.MeshBasicMaterial({ color: isReturn ? 0x3b4d68 : 0x5aa9e6, transparent: true, opacity: isReturn ? 0.5 : 0.75 }),
        );
        this.scene.add(tube);
        this.links.push(tube);
        const count = isReturn ? 2 : 5;
        for (let k = 0; k < count; k++) {
          const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(isReturn ? 0.06 : 0.09, 12, 8),
            new THREE.MeshBasicMaterial({ color: isReturn ? 0x9fb3cc : 0xbfe6ff }),
          );
          this.scene.add(mesh);
          this.particles.push({ mesh, curve: this.curves.length - 1, t: k / count });
        }
      }
    }
  }

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    if (!this.renderer) return;
    const dt = Math.min(0.05, this.clock.getDelta());
    const speed = (this.busy ? 0.55 : 0.12) * (this.reduced ? 0.3 : 1);
    for (const p of this.particles) {
      p.t = (p.t + dt * speed) % 1;
      p.mesh.position.copy(this.curves[p.curve].getPointAt(p.t));
    }
    const t = this.clock.elapsedTime;
    for (const n of this.nodes) {
      const load = (n.view.cpu ?? 0) / 100;
      const pulse = 1 + (this.reduced ? 0 : 0.04 * Math.sin(t * (1.5 + 3 * load)));
      n.core.scale.setScalar(pulse);
      (n.core.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.35 + 0.65 * load;
      (n.halo.material as THREE.MeshBasicMaterial).opacity = 0.05 + 0.12 * load;
    }
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);

    // project labels
    const w = this.renderer.domElement.clientWidth;
    const h = this.renderer.domElement.clientHeight;
    for (const n of this.nodes) {
      const p = n.group.position.clone();
      p.y += 1.3;
      p.project(this.camera);
      const behind = p.z > 1;
      n.label.style.display = behind ? 'none' : '';
      n.label.style.transform = `translate(-50%, -100%) translate(${((p.x + 1) / 2) * w}px, ${((1 - p.y) / 2) * h}px)`;
    }
  };
}
