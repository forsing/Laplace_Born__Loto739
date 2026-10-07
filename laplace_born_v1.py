# Potreban je NumPy ≥ 2.0
# scipy



import csv
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


CSV_PATH = Path(
    "data/loto7_4698_k80.csv"
    # "/data/loto7_4698_k80_loto_2971.csv"
    # "/data/loto7_4698_k80_loto_plus_1727.csv"
)

N = 39
K = 7
TOTAL = math.comb(N, K)

BATCH = 50_000
MAX_ITER = 300
TOL = 1e-6

PAIRS = list(itertools.combinations(range(N), 2))

PAIR_ID = {
    pair: 1 + N + index
    for index, pair in enumerate(PAIRS)
}

FEATURES = 1 + N + len(PAIRS)

SINGLE_RATIO = N / K
PAIR_RATIO = math.comb(N, 2) / math.comb(K, 2)

# Simetrični prior za regularizaciju naučenih distribucija.
# Ne koristi se kao zamenski prediktor.
BASE = np.concatenate((
    [0.5],
    np.full(N, 0.25 / N),
    np.full(len(PAIRS), 0.25 / len(PAIRS)),
))


def load_csv(path):
    """
    CSV bez zaglavlja, sedam brojeva po redu.
    Prvi red najstariji, poslednji najnoviji.
    Redosled izvlačenja se ne menja.
    """
    draws = []

    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for line, row in enumerate(csv.reader(file), 1):
            if not row or all(not value.strip() for value in row):
                continue

            try:
                draw = sorted(int(value.strip()) for value in row)
            except ValueError as exc:
                raise ValueError(
                    f"Red {line}: neispravan ceo broj."
                ) from exc

            if (
                len(draw) != K
                or len(set(draw)) != K
                or not all(1 <= value <= N for value in draw)
            ):
                raise ValueError(
                    f"Red {line}: potrebno je sedam "
                    "različitih brojeva od 1 do 39."
                )

            draws.append(draw)

    if len(draws) < 50:
        raise ValueError("Potrebno je najmanje 50 izvlačenja.")

    return np.asarray(draws, dtype=np.int64)


def features(draws):
    """
    Predstavlja cele sedmorke pomoću normalizovanih komponenti
    zajedničke distribucije.

    Svako skriveno stanje uči svoju mešavinu:
    - osnovne komponente;
    - komponenti sedmorki koje sadrže određeni broj;
    - komponenti sedmorki koje sadrže određeni par.

    Vrednosti su odnosi prema osnovnoj distribuciji.
    """
    indices = []
    values = []
    indptr = [0]

    for draw in draws:
        numbers = draw - 1

        indices.append(0)
        values.append(1.0)

        for number in numbers:
            indices.append(1 + int(number))
            values.append(SINGLE_RATIO)

        for a, b in itertools.combinations(numbers, 2):
            indices.append(PAIR_ID[(int(a), int(b))])
            values.append(PAIR_RATIO)

        indptr.append(len(indices))

    matrix = csr_matrix(
        (values, indices, indptr),
        shape=(len(draws), FEATURES),
    )

    if not np.allclose(matrix @ BASE, 1.0, atol=1e-12):
        raise RuntimeError("Neispravna normalizacija komponenti.")

    return matrix


def forward_backward(emission, initial, transition):
    length, states = emission.shape

    forward = np.empty_like(emission)
    scale = np.empty(length)

    forward[0] = initial * emission[0]
    scale[0] = forward[0].sum()
    forward[0] /= scale[0]

    for t in range(1, length):
        forward[t] = (
            forward[t - 1] @ transition
        ) * emission[t]

        scale[t] = forward[t].sum()
        forward[t] /= scale[t]

    backward = np.ones_like(emission)

    for t in range(length - 2, -1, -1):
        backward[t] = (
            transition
            @ (emission[t + 1] * backward[t + 1])
            / scale[t + 1]
        )

    gamma = forward * backward
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((states, states))

    for t in range(length - 1):
        xi += (
            forward[t, :, None]
            * transition
            * (emission[t + 1] * backward[t + 1])[None, :]
            / scale[t + 1]
        )

    likelihood = float(np.log(scale).sum())

    return gamma, xi, likelihood, forward[-1]


def initialize(matrix, draws, states, tau):
    """
    Deterministička inicijalizacija po zbiru sedmorke.
    Zbir služi samo za početnu podelu skrivenih stanja.
    Hronološki redosled podataka ostaje sačuvan.
    """
    ordering = np.argsort(draws.sum(axis=1), kind="stable")
    phi = []

    for group in np.array_split(ordering, states):
        counts = (
            BASE
            * np.asarray(matrix[group].sum(axis=0)).ravel()
        )

        weights = counts + tau * BASE
        phi.append(weights / weights.sum())

    phi = np.asarray(phi)
    initial = np.full(states, 1.0 / states)

    if states == 1:
        transition = np.ones((1, 1))
    else:
        transition = np.full(
            (states, states),
            0.2 / (states - 1),
        )
        np.fill_diagonal(transition, 0.8)

    return phi, initial, transition


def fit(matrix, draws, states, strength):
    """
    EM učenje zajedničkih distribucija skrivenih stanja
    i verovatnoća prelaska između njih.

    Nema slučajne inicijalizacije ni slučajnog uzorkovanja.
    """
    tau = strength * len(draws) / states

    phi, initial, transition = initialize(
        matrix, draws, states, tau
    )

    previous = None
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        emission = np.asarray(matrix @ phi.T)

        gamma, xi, likelihood, _ = forward_backward(
            emission,
            initial,
            transition,
        )

        objective = (
            likelihood
            + tau * float(
                (BASE[None, :] * np.log(phi)).sum()
            )
            + float(np.log(transition).sum())
            + float(np.log(initial).sum())
        )

        if previous is not None:
            if objective < (
                previous
                - 1e-6 * max(1.0, abs(previous))
            ):
                raise RuntimeError(
                    "EM cilj je opao; model nije prihvaćen."
                )

            if abs(objective - previous) <= (
                TOL * max(1.0, abs(previous))
            ):
                converged = True
                break

        previous = objective

        # Očekivani doprinosi latentnih komponenti distribucije.
        counts = phi * np.asarray(
            matrix.T @ (gamma / emission)
        ).T

        phi = counts + tau * BASE[None, :]
        phi /= phi.sum(axis=1, keepdims=True)

        initial = gamma[0] + 1.0
        initial /= initial.sum()

        transition = xi + 1.0
        transition /= transition.sum(axis=1, keepdims=True)

    if (
        not np.all(np.isfinite(phi))
        or not np.allclose(phi.sum(axis=1), 1.0)
    ):
        raise RuntimeError("Neispravna distribucija stanja.")

    return {
        "phi": phi,
        "initial": initial,
        "transition": transition,
        "iterations": iteration,
        "converged": converged,
    }


def filter_history(matrix, model):
    """Procena trenutnog stanja nakon poznate istorije."""
    emission = np.asarray(matrix @ model["phi"].T)
    posterior = model["initial"].copy()

    for t, row in enumerate(emission):
        if t:
            posterior = posterior @ model["transition"]

        posterior *= row
        posterior /= posterior.sum()

    return posterior


def validation_score(train_matrix, validation_matrix, model):
    """
    Svako kasnije izvlačenje predviđa se pre nego što
    se njegov rezultat koristi za ažuriranje procene stanja.
    """
    posterior = filter_history(train_matrix, model)
    emission = np.asarray(validation_matrix @ model["phi"].T)
    score = 0.0

    for row in emission:
        predicted = posterior @ model["transition"]
        ratio = float(predicted @ row)

        score += math.log(ratio) - math.log(TOTAL)
        posterior = predicted * row / ratio

    return score / len(emission)


def select_model(draws):
    """
    Bira najbolji HMM među šest determinističkih kandidata.
    Nijedna referentna distribucija ne zamenjuje HMM.
    """
    split = int(0.8 * len(draws))

    training = draws[:split]
    validation = draws[split:]

    train_matrix = features(training)
    validation_matrix = features(validation)

    best_score = -math.inf
    best = None

    print(
        f"Obuka: {split}; "
        f"hronološka provera: {len(validation)}",
        flush=True,
    )

    for states in (1, 2, 3):
        for strength in (0.02, 0.2):
            model = fit(
                train_matrix,
                training,
                states,
                strength,
            )

            score = validation_score(
                train_matrix,
                validation_matrix,
                model,
            )

            status = (
                "konvergirao"
                if model["converged"]
                else "limit iteracija"
            )

            print(
                f"Stanja={states}, "
                f"regularizacija={strength}, "
                f"log P={score:.9f}, {status}",
                flush=True,
            )

            if score > best_score + 1e-10:
                best_score = score
                best = (states, strength)

    return best


def combined_weights(draws, selected):
    if selected is None:
        raise RuntimeError("Nije izabran nijedan HMM model.")

    matrix = features(draws)
    states, strength = selected

    # Završna obuka koristi CEO ažurni CSV.
    model = fit(matrix, draws, states, strength)

    posterior = filter_history(matrix, model)
    next_state = posterior @ model["transition"]

    # Distribucija sledeće sedmorke integrisana preko
    # svih mogućih sledećih skrivenih stanja.
    weights = next_state @ model["phi"]

    print(
        f"Završni model: {states} stanja; "
        f"regularizacija={strength}; "
        f"svih {len(draws)} izvlačenja.",
        flush=True,
    )

    print(
        "Verovatnoće sledećeg stanja:",
        " ".join(f"{value:.6f}" for value in next_state),
        flush=True,
    )

    if not model["converged"]:
        print(
            f"Završna obuka dostigla je limit "
            f"od {MAX_ITER} iteracija.",
            flush=True,
        )

    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Neispravna završna normalizacija.")

    return weights


def global_maximum(weights):
    """
    Tačna globalna pretraga svih validnih sedmorki.
    Bira maksimum zajedničke prediktivne distribucije.

    Kod jednakih maksimuma ostaje leksikografski prva.
    """
    singles = SINGLE_RATIO * weights[1:1 + N]
    pairs = np.zeros((N, N))

    for (a, b), weight in zip(PAIRS, weights[1 + N:]):
        pairs[a, b] = pairs[b, a] = PAIR_RATIO * weight

    positions = list(itertools.combinations(range(K), 2))
    iterator = itertools.combinations(range(N), K)

    best = None
    best_ratio = -math.inf
    examined = 0
    ratio_sum = 0.0

    while True:
        batch = list(itertools.islice(iterator, BATCH))

        if not batch:
            break

        combinations = np.asarray(batch, dtype=np.int64)

        ratios = (
            weights[0]
            + singles[combinations].sum(axis=1)
        )

        for a, b in positions:
            ratios += pairs[
                combinations[:, a],
                combinations[:, b],
            ]

        index = int(np.argmax(ratios))
        value = float(ratios[index])

        if value > best_ratio:
            best_ratio = value
            best = tuple(
                int(number) + 1
                for number in combinations[index]
            )

        examined += len(batch)
        ratio_sum += float(ratios.sum())

    if examined != TOTAL or best is None:
        raise RuntimeError("Globalna pretraga nije završena.")

    if not np.isclose(ratio_sum, TOTAL, rtol=1e-9):
        raise RuntimeError("Distribucija nije normalizovana.")

    return best, best_ratio / TOTAL


def main():
    draws = load_csv(CSV_PATH)

    print("Laplace-Born Loto 7/39 — v1", flush=True)
    print(
        f"CSV: {CSV_PATH}; izvlačenja: {len(draws)}",
        flush=True,
    )
    print(
        "Prvi red najstariji; poslednji najnoviji.",
        flush=True,
    )

    selected = select_model(draws)
    weights = combined_weights(draws, selected)

    print(
        f"Pregled svih {TOTAL} kombinacija...",
        flush=True,
    )

    prediction, probability = global_maximum(weights)

    print("\nNEXT:", " ".join(map(str, prediction)))
    print(f"Verovatnoća po modelu: {probability:.12g}")


if __name__ == "__main__":
    main()



"""
Laplace-Born Loto 7/39 — v1
CSV: /data/loto7_4698_k80.csv; izvlačenja: 4698
Prvi red najstariji; poslednji najnoviji.
Obuka: 3758; hronološka provera: 940
Stanja=1, regularizacija=0.02, log P=-16.571142863, konvergirao
Stanja=1, regularizacija=0.2, log P=-16.547585150, konvergirao
Stanja=2, regularizacija=0.02, log P=-16.587628051, limit iteracija
Stanja=2, regularizacija=0.2, log P=-16.549941807, konvergirao
Stanja=3, regularizacija=0.02, log P=-16.600020137, limit iteracija
Stanja=3, regularizacija=0.2, log P=-16.553667921, konvergirao
Završni model: 1 stanja; regularizacija=0.2; svih 4698 izvlačenja.
Verovatnoće sledećeg stanja: 1.000000
Pregled svih 15380937 kombinacija...

NEXT: 8 x 26 y 33 z 35
Verovatnoća po modelu: 8.55019348651e-08





Laplace-Born Loto 7/39 — v1
CSV: /data/loto7_4698_k80_loto_2971.csv; izvlačenja: 2971
Prvi red najstariji; poslednji najnoviji.
Obuka: 2376; hronološka provera: 595
Stanja=1, regularizacija=0.02, log P=-16.573649630, konvergirao
Stanja=1, regularizacija=0.2, log P=-16.549593868, konvergirao
Stanja=2, regularizacija=0.02, log P=-16.616604321, limit iteracija
Stanja=2, regularizacija=0.2, log P=-16.554698887, konvergirao
Stanja=3, regularizacija=0.02, log P=-16.621567941, limit iteracija
Stanja=3, regularizacija=0.2, log P=-16.558990740, konvergirao
Završni model: 1 stanja; regularizacija=0.2; svih 2971 izvlačenja.
Verovatnoće sledećeg stanja: 1.000000
Pregled svih 15380937 kombinacija...

NEXT: 5 x 11 y 23 z 33
Verovatnoća po modelu: 9.60381768448e-08





Laplace-Born Loto 7/39 — v1
CSV: /data/loto7_4698_k80_loto_plus_1727.csv; izvlačenja: 1727
Prvi red najstariji; poslednji najnoviji.
Obuka: 1381; hronološka provera: 346
Stanja=1, regularizacija=0.02, log P=-16.592125652, konvergirao
Stanja=1, regularizacija=0.2, log P=-16.547902135, konvergirao
Stanja=2, regularizacija=0.02, log P=-16.630583226, limit iteracija
Stanja=2, regularizacija=0.2, log P=-16.556991701, konvergirao
Stanja=3, regularizacija=0.02, log P=-16.703075007, limit iteracija
Stanja=3, regularizacija=0.2, log P=-16.564698196, konvergirao
Završni model: 1 stanja; regularizacija=0.2; svih 1727 izvlačenja.
Verovatnoće sledećeg stanja: 1.000000
Pregled svih 15380937 kombinacija...

NEXT: 8 x 23 y 31 z 37
Verovatnoća po modelu: 1.11039263263e-07
"""





"""
Korisna je ideja „stanja” kao sažetka informacija potrebnih za predviđanje sledećeg izvlačenja. 
U lotou 7/39 to može da bude osnova modela, bez pretpostavke da daje gotovo rešenje.
Laplasovo stanje predstavlja potpun fizički opis sistema, iz kojeg bi deterministička dinamika određivala budućnost. 

Za tvoj model korisne su ove ideje:
- Skriveno stanje sistema. Bubanj, kuglice i način izvlačenja mogu imati svojstva koja ne vidiš u CSV-u. Istorija izvlačenja tada služi da procenjuješ moguća skrivena stanja.
- Distribucija umesto jedne tačke. Stanje modela može biti cela zajednička distribucija sedmorki, sa zavisnostima među brojevima, umesto prosečne kombinacije.
- Promena stanja kroz vreme. Ceo CSV može služiti za učenje kako se obrasci menjaju, dok redosled izvlačenja pomaže da proceniš trenutno stanje. Korišćenje svih redova ne mora značiti da svi opisuju isto stanje.
- Ažuriranje posle izvlačenja. Novi rezultat menja procenu skrivenog stanja, a zatim i distribuciju sledeće sedmorke.

Ovde je korisna ideja da ne procenjujem samo „koji brojevi izlaze”, nego „koje stanje sistema najbolje objašnjava dosadašnja izvlačenja i kakvu distribuciju sledeće sedmorke ono daje”. 
Projekat zasnivam samo na ideji stanja sistema kod Laplasa i Borna, primenjenoj na predikciju lotoa 7/39.
Skriveni Markovljev model sa zajedničkom distribucijom sedmorki u svakom stanju. To je konkretan način da ideju „stanja sistema” pretvorim u prediktor za 7/39.

Model bi imao tri dela:
1. Skrivena stanja. Svako stanje predstavlja drugačiji statistički obrazac izvlačenja. Model iz celog CSV-a uči te obrasce i verovatnoće prelaska između njih. Ne pretpostavljam da pronađeno stanje odgovara nekom poznatom fizičkom stanju bubnja.
2. Distribucija unutar stanja. Svako stanje daje zajedničku distribuciju nad kombinacijama od tačno sedam različitih brojeva. Predlažem fleksibilnu distribuciju sa pojedinačnim i parnim interakcijama, bez nametanja normalne raspodele i bez prostog rangiranja najčešćih brojeva.
3. Predikcija sledećeg izvlačenja. Na osnovu istorije procenjujemo trenutno stanje, zatim moguće sledeće stanje, pa kombinujemo njihove distribucije:

Rezultat je jedna sedmorka sa najvećom verovatnoćom u toj prediktivnoj distribuciji, uz determinističko pravilo za izjednačenja.
Broj stanja i jačinu regularizacije biram hronološkom proverom: obuka na ranijim redovima, procena na kasnijim. Uključujemo i model sa jednim stanjem, da više stanja ostane samo ako poboljšava predikciju. Završno učenje koristi ceo CSV; nema random uzorkovanja.

Laplas daje koncept skrivenog stanja i njegovog razvoja, 
Born koncept distribucije ishoda uslovljene stanjem.


Provereni su učenje stanja, hronološka validacija, normalizacija i pretraga svih 15.380.937 kombinacija na kontrolnim podacima.
Kod koristi skriveni Markovljev model sa zajedničkim distribucijama sedmorki, bez normalne raspodele i random postupaka.
Ako uniformna distribucija pobedi na proveri, kod to jasno ispisuje; tada nema podržane prednosti neke sedmorke.


HMM uvek daje jednu predikciju iz CSV-a, birajući najbolji provereni model sa jednim, dva ili tri stanja.


Koristi zajedničku distribuciju sedmorki i determinističko učenje. 
Nema random uzorkovanja, rangiranja najčešćih brojeva niti zamene predikcije uniformnom raspodelom. 
Hronološka provera izabrala je model sa jednim stanjem kao najbolji među šest proverenih HMM modela.
"""
