> Revisado em 11/09/2026. Quati está inativa atualmente, por confirmação do
> responsável. Os caminhos e incidentes desse host abaixo são registros da
> preparação anterior. O procedimento e a identidade do recorte permanecem
> úteis; sua menção não indica uma campanha em execução. Os manifests não
> são imagens nem são entregues pelo clone: confirme sua origem e hash antes
> de preparar outro host. Confira os caminhos no mapa operacional antes de usar.

# Download e setup do ImageNet-100

Este guia registra o procedimento reproduzível para preparar o ImageNet-100
usado neste repositório. O processo tem duas entradas distintas:

1. as imagens do ILSVRC/ImageNet, obtidas pela competição oficial do Kaggle;
2. os manifests do recorte de 100 classes usado pelo PyCIL.

O ZIP dos manifests **não contém imagens** e não substitui o download do
ILSVRC. Não versionar imagens, tokens, arquivos extraídos ou credenciais no
Git. O acesso ao ImageNet está sujeito aos termos do provedor e deve ser feito
por uma conta autorizada.

## Layout esperado pelo TagFex

O carregador `ImageNet100` em `modules/data/dataset.py` usa `ImageFolder` e
espera esta estrutura:

```text
<dataset_root>/
├── train/
│   ├── n01514668/
│   │   └── *.JPEG
│   └── ...                         # 100 diretórios de synsets
└── val/
    ├── n01514668/
    │   └── *.JPEG
    └── ...                         # os mesmos 100 synsets
```

Os YAMLs devem apontar `dataset_root` para esse diretório. A configuração sem
FFCV lê as imagens diretamente. Os cenários `imagenet100_ffcv*.yaml` também
exigem os arquivos `.beton` indicados por `train_beton_path` e
`val_beton_path`; preparar somente o layout acima não cria esses arquivos.

## Requisitos de armazenamento e ambiente

- usar um filesystem com algumas centenas de GB livres para o arquivo de
  aproximadamente 155 GB e sua extração;
- evitar `/home` em hosts nos quais esse volume seja pequeno;
- manter download, extração e dataset final no mesmo filesystem quando se
  pretender usar hard links;
- usar Python 3.10 ou superior em um ambiente virtual isolado.

Exemplo genérico, substituindo `<VOLUME_GRANDE>` pelo volume do host:

```bash
python3 -m venv <VOLUME_GRANDE>/tools/kagglehub-venv
<VOLUME_GRANDE>/tools/kagglehub-venv/bin/python -m pip install --upgrade pip
<VOLUME_GRANDE>/tools/kagglehub-venv/bin/python -m pip install 'kagglehub>=1.0.2'
```

A versão `1.0.2` foi validada na Quati em 1 de setembro de 2026.

## Autorizar a conta Kaggle

Primeiro, acessar a página da competição e aceitar suas regras com a mesma
conta que gerou o token:

```text
https://www.kaggle.com/competitions/imagenet-object-localization-challenge
```

Criar um novo token no Kaggle. Para tokens no formato `KGAT_...`, armazená-lo
sem exibi-lo no terminal:

```bash
install -d -m 700 ~/.kaggle
umask 077
read -rsp "Cole o token KGAT: " IMAGENET_KAGGLE_TOKEN
printf '\n'
printf '%s\n' "$IMAGENET_KAGGLE_TOKEN" > ~/.kaggle/access_token
unset IMAGENET_KAGGLE_TOKEN
chmod 600 ~/.kaggle/access_token
```

Validar apenas presença, formato e permissões; nunca imprimir o conteúdo:

```bash
test -s ~/.kaggle/access_token
grep -q '^KGAT_' ~/.kaggle/access_token
stat -c 'arquivo=%n permissao=%a proprietario=%U bytes=%s' \
  ~/.kaggle/access_token
```

O resultado deve indicar permissão `600`. Em um host remoto, transferir a
credencial somente por um canal autorizado e restaurar essa permissão.

## Baixar o ILSVRC pelo KaggleHub

Definir um destino explícito é obrigatório para não usar inadvertidamente o
cache padrão em `/home`. Exemplo da Quati:

```python
import kagglehub

path = kagglehub.competition_download(
    "imagenet-object-localization-challenge",
    output_dir=(
        "/mnt/raid/home/tiago/data/downloads/"
        "imagenet-object-localization-challenge"
    ),
)
print("Path to competition files:", path)
```

Na Quati, o CLI `kaggle competitions download` retornou `403`, enquanto a
mesma conta e o mesmo token funcionaram com `kagglehub`. Portanto, preferir o
KaggleHub neste fluxo; um `403` ainda pode indicar regras não aceitas, token de
outra conta ou token inválido.

Para downloads longos, salvar o trecho acima como `download_imagenet.py` no
volume grande e executá-lo como unidade persistente. Antes, habilitar o
gerenciador de usuário após o logout:

```bash
loginctl enable-linger "$USER"
```

Exemplo da Quati:

```bash
systemd-run --user \
  --unit=imagenet-kagglehub-download-<AAAAMMDD> \
  --collect \
  --property=StandardOutput=append:/mnt/raid/home/tiago/data/downloads/imagenet-kagglehub-download.log \
  --property=StandardError=append:/mnt/raid/home/tiago/data/downloads/imagenet-kagglehub-download.log \
  /mnt/raid/home/tiago/tools/kaggle-cli-py312/bin/python \
  /mnt/raid/home/tiago/data/downloads/download_imagenet.py
```

Monitorar sem depender da sessão SSH:

```bash
systemctl --user status imagenet-kagglehub-download-<AAAAMMDD>.service
du -sh <VOLUME_GRANDE>/data/downloads/imagenet-object-localization-challenge
tail -n 30 <VOLUME_GRANDE>/data/downloads/imagenet-kagglehub-download.log
df -h <VOLUME_GRANDE>
```

Não usar `force_download=True` nem apagar um `.archive` parcial antes de
classificar o estado do serviço e verificar se o cliente pode reaproveitá-lo.

## Obter e validar os manifests do ImageNet-100

O arquivo validado neste projeto é:

```text
data/datasets/imagenet100/ImageNet100-20260901T134946Z-1-001.zip
```

SHA-256:

```text
dd04279da934cd92d79f7bf3bbf7d0cc2cc5e04b4e35cbbcf7a811e9dcd487ea
```

Ele contém `ImageNet100/train.txt` e `ImageNet100/eval.txt`. Cada bloco começa
com o synset e é seguido pelos caminhos relativos de suas imagens. A auditoria
local confirmou:

- 100 classes únicas, na mesma ordem em treino e validação;
- 128.856 caminhos únicos de treino;
- 5.000 caminhos únicos de validação, 50 por classe;
- nenhum caminho compartilhado entre os dois splits;
- todos os nomes terminam em `.JPEG` ou `.JPG`.

Validar e extrair em qualquer novo host:

```bash
sha256sum ImageNet100-20260901T134946Z-1-001.zip
unzip -t ImageNet100-20260901T134946Z-1-001.zip
unzip ImageNet100-20260901T134946Z-1-001.zip -d <MANIFEST_ROOT>
```

## Extrair o ILSVRC e materializar o recorte

Após o KaggleHub terminar, inventariar os artefatos antes de extrair. A
estrutura e o nome do arquivo interno podem variar entre versões do pacote:

```bash
find <DOWNLOAD_ROOT> -maxdepth 3 -type f -printf '%s %p\n' | sort -n
```

Localizar o diretório `ILSVRC/Data/CLS-LOC`. Adotar então:

```text
<SOURCE_TRAIN> = .../ILSVRC/Data/CLS-LOC/train
<SOURCE_VAL>   = .../ILSVRC/Data/CLS-LOC/val
```

No treino oficial, as imagens normalmente já ficam sob o diretório do synset.
A validação pode estar em um único diretório. Para cada caminho
`<synset>/<arquivo>` dos manifests:

- treino: selecionar `<SOURCE_TRAIN>/<synset>/<arquivo>`;
- validação: selecionar `<SOURCE_VAL>/<arquivo>` se a origem for plana, ou
  `<SOURCE_VAL>/<synset>/<arquivo>` se já estiver organizada;
- destino: criar `train/<synset>/...` ou `val/<synset>/...`.

Preferir hard links quando origem e destino estiverem no mesmo filesystem;
eles evitam duplicar as imagens e não quebram se o diretório de origem for
renomeado. Se estiverem em filesystems diferentes, usar links simbólicos com
origem estável ou cópia. Não misturar estratégias sem registrar a decisão.

Exemplo reproduzível com hard links. Revisar os quatro caminhos antes de
executar; o código ignora as linhas que contêm somente o synset, aceita uma
retomada se o destino já for o mesmo inode e interrompe em qualquer colisão:

```bash
set -euo pipefail

MANIFEST_ROOT="/caminho/para/manifests/ImageNet100"
SOURCE_TRAIN="/caminho/para/ILSVRC/Data/CLS-LOC/train"
SOURCE_VAL="/caminho/para/ILSVRC/Data/CLS-LOC/val"
DATASET_ROOT="/caminho/para/data/datasets/imagenet100"

while IFS= read -r rel; do
  case "$rel" in
    */*) ;;
    *) continue ;;
  esac
  src="$SOURCE_TRAIN/$rel"
  dst="$DATASET_ROOT/train/$rel"
  test -f "$src" || { echo "Treino ausente: $src" >&2; exit 1; }
  install -d "$(dirname "$dst")"
  if test -e "$dst"; then
    test "$src" -ef "$dst" || { echo "Colisao: $dst" >&2; exit 1; }
  else
    ln "$src" "$dst"
  fi
done < "$MANIFEST_ROOT/train.txt"

while IFS= read -r rel; do
  case "$rel" in
    */*) ;;
    *) continue ;;
  esac
  src="$SOURCE_VAL/$rel"
  test -f "$src" || src="$SOURCE_VAL/${rel#*/}"
  dst="$DATASET_ROOT/val/$rel"
  test -f "$src" || { echo "Validacao ausente: $rel" >&2; exit 1; }
  install -d "$(dirname "$dst")"
  if test -e "$dst"; then
    test "$src" -ef "$dst" || { echo "Colisao: $dst" >&2; exit 1; }
  else
    ln "$src" "$dst"
  fi
done < "$MANIFEST_ROOT/eval.txt"
```

Se hard links não forem possíveis, copiar o bloco para um script revisado e
trocar somente `ln "$src" "$dst"` por `ln -s "$(realpath "$src")" "$dst"`
ou `cp --reflink=auto "$src" "$dst"`. Não fazer substituição textual cega em
um terminal.

Antes da materialização completa, testar uma imagem de treino e uma de
validação do primeiro synset (`n01514668`) com Pillow. Depois, conferir:

```bash
find <DATASET_ROOT>/train -mindepth 1 -maxdepth 1 -type d | wc -l
find <DATASET_ROOT>/val   -mindepth 1 -maxdepth 1 -type d | wc -l
find <DATASET_ROOT>/train \( -type f -o -type l \) | wc -l
find <DATASET_ROOT>/val   \( -type f -o -type l \) | wc -l
```

Os resultados esperados são, respectivamente, `100`, `100`, `128856` e
`5000`. Confirmar também que não existem links quebrados:

```bash
find -L <DATASET_ROOT>/train <DATASET_ROOT>/val -type l -print -quit
```

Por fim, validar com o próprio carregador do repositório antes de criar uma
fila. A classe deve enxergar 100 classes e os mesmos totais:

```python
from modules.data.dataset import ImageNet100

root = "<DATASET_ROOT>"
train = ImageNet100(root=root, split="train")
val = ImageNet100(root=root, split="val")

assert len(train.classes) == len(val.classes) == 100
assert train.classes == val.classes
assert len(train) == 128_856
assert len(val) == 5_000
print("ImageNet-100 validado")
```

Só iniciar experimentos depois de registrar o caminho efetivo do dataset no
overlay específico do host e carregar ao menos uma amostra real de cada split.

## Checklist de migração para outro host

1. Confirmar espaço no volume de download, extração, dataset e logs.
2. Instalar `kagglehub` em venv isolado nesse volume.
3. Aceitar as regras com a conta correta e instalar o token com modo `600`.
4. Usar `output_dir` explícito; nunca depender do cache padrão.
5. Executar como serviço persistente e registrar o log fora de `/home`.
6. Validar conclusão e integridade antes de remover qualquer arquivo parcial.
7. Validar o SHA-256 e os totais dos manifests.
8. Materializar exatamente os caminhos listados, preservando os synsets.
9. Confirmar `100/100` classes e `128856/5000` imagens.
10. Testar `ImageNet100` e uma amostra real antes de enfileirar experimentos.
