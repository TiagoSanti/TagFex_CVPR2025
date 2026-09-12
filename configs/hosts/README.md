# Perfis operacionais

Perfis em `<host>/<campanha>/` contêm somente `log_dir`, `dataset_root` e/ou
`ckpt_dir`. Controles científicos das campanhas migradas ficam em
`experiments/<campanha>/configs/`.

A composição de uma fila é: base científica → controles comuns → variante →
overlay opcional de dataset → armazenamento do host. Preserve a ordem.
Os perfis não são aplicados automaticamente pelo hostname: a fila declara
explicitamente cada arquivo. `profile_path.py` permite que o shell leia o mesmo
destino sem duplicar seu valor.

Os caminhos de Wolverine são referências ao filesystem remoto e podem ser
inspecionados em Xavier sem acessá-los. O CUB da campanha central vem da campanha
`ANT_detach_cub20_20260904`; não remova esse dataset pelo fato de a campanha
anterior ter terminado.

Esta separação cobre três campanhas recentes. Os caminhos antigos continuam
como links simbólicos; YAMLs legados ainda não migrados permanecem autocontidos.
Veja os [contratos e limites da etapa 2](../../docs/RESTRUCTURING_STAGE2.md).
