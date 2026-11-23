from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.agent import AgentConfig
from src.config.run import DataConfig, RunConfig
from src.config.training import TrainingConfig
from src.config.values import _expand_env_vars, read_config_mapping


@dataclass(frozen=True)
class AppConfig:
    agent: AgentConfig | None
    run: RunConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        required: bool = False,
        load_agent: bool = True,
    ) -> "AppConfig":
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
                )
            return cls(
                agent=AgentConfig.from_file(path, required=required) if load_agent else None,
                run=RunConfig(), data=DataConfig(),
            )

        raw = read_config_mapping(path, expand_env=False)
        agent_settings = raw.get("agent")
        if load_agent and not isinstance(agent_settings, dict):
            raise ValueError('Config key "agent" must contain a mapping')
        run_settings = raw.get("run", {})
        if run_settings is None:
            run_settings = {}
        if not isinstance(run_settings, dict):
            raise ValueError('Config key "run" must contain a mapping')
        data_settings = raw.get("data", {})
        if data_settings is None:
            data_settings = {}
        if not isinstance(data_settings, dict):
            raise ValueError('Config key "data" must contain a mapping')
        run_settings = _expand_env_vars(run_settings)
        data_settings = _expand_env_vars(data_settings)
        return cls(
            agent=(
                AgentConfig.from_file(path, required=required)
                if load_agent
                else None
            ),
            run=RunConfig.from_mapping(run_settings),
            data=DataConfig.from_mapping(data_settings),
            training=TrainingConfig.from_mapping(_expand_env_vars(raw.get("training"))),
        )
