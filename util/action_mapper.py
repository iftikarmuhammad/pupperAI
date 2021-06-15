ENV_ID_TO_POLICY = {
    'gallop': ('rex_gym/policies/galloping/balanced', 'model.ckpt-20000000'),
    'walk': ('rex_gym/policies/walking/alternating_legs/pupper-mix-2m-9/', 'model.ckpt-8000000'),
    # 'standup': ('rex_gym/policies/standup', 'model.ckpt-10000000'),
    'standup': ('rex_gym/policies/standup/pupper-500k/', 'model.ckpt-1000000'),
    'turn': ('rex_gym/policies/turn', 'model.ckpt-16000000')
}

ENV_ID_TO_ENV_NAMES = {
    'gallop': 'RexReactiveEnv',
    'walk': 'RexWalkEnv',
    'turn': 'RexTurnEnv',
    'standup': 'RexStandupEnv'
}
