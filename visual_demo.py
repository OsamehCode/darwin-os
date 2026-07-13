"""Optional Pygame visual demo. Auto-fallback to headless when no display."""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("DISPLAY") and not _is_macos():
        print("[visual_demo] no $DISPLAY detected — falling back to headless demo.")
        import headless_demo
        headless_demo.main()
        return 0

    import pygame  # type: ignore
    from darwin_os.controller import Controller
    from darwin_os.crisis_detector import CrisisDetector
    from darwin_os.digital_twin import DigitalTwin
    from darwin_os.environment import Vec2
    from darwin_os.evolution import EvolutionConfig, EvolutionEngine
    from darwin_os.scenarios import surprise_vortex_world
    from darwin_os.verifier import SafetyVerifier, default_rules

    env = surprise_vortex_world(seed=42, vortex_t_start=6.0)
    twin = DigitalTwin(env)
    engine = EvolutionEngine(twin=twin, verifier=SafetyVerifier(default_rules()),
                              config=EvolutionConfig(population_size=80,
                                                     n_generations=12, seed=7))
    ctrl = Controller(twin=twin, engine=engine)
    detector = CrisisDetector(twin=twin, verifier=SafetyVerifier(default_rules()),
                              lookahead_steps=20)

    W, H = 600, 600
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DARWIN-OS")
    clock = pygame.time.Clock()

    def to_px(p):
        return int(p[0] / env.arena_size * W), int(p[1] / env.arena_size * H)

    s = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        a = ctrl.act(s)
        if detector.assess(ctrl, s).triggered:
            ctrl.evolve_and_swap(t=env.time, reason="visual-demo")
        s = env.step(a["thrust"], a["yaw_rate"])
        screen.fill((255, 255, 255))
        for ob in env.obstacles:
            pygame.draw.circle(screen, (200, 50, 50), to_px((ob.center.x, ob.center.y)),
                                int(ob.radius / env.arena_size * H))
        if env.vortex:
            cx, cy = to_px((env.vortex.center.x, env.vortex.center.y))
            pygame.draw.circle(screen, (180, 0, 180), (cx, cy),
                                int(env.vortex.radius / env.arena_size * H), 2)
        pygame.draw.circle(screen, (30, 144, 255), to_px((env.goal.x, env.goal.y)),
                            int(3 / env.arena_size * H))
        ax, ay = to_px((s.agent.position.x, s.agent.position.y))
        pygame.draw.circle(screen, (0, 0, 0), (ax, ay),
                            int(s.agent.radius / env.arena_size * H))
        pygame.display.flip()
        clock.tick(15)
    pygame.quit()
    return 0


def _is_macos() -> bool:
    return sys.platform == "darwin"


if __name__ == "__main__":
    sys.exit(main())
