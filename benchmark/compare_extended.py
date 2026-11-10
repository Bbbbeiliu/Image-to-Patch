import copy
from dataclasses import replace
import compare
COMMON_KEYS = ('input', 'interpolation', 'evaluation', 'empty_policy')
original_validate = compare.validate_runs
original_write = compare.write_outputs

def common_protocol(run):
    protocol = run.metadata['protocol']
    if not isinstance(protocol, dict) or any((k not in protocol for k in COMMON_KEYS)):
        raise ValueError('Extended comparison requires explicit common evaluation keys')
    return {key: protocol[key] for key in COMMON_KEYS}

def validate(runs):
    normalized = []
    for run in runs:
        metadata = copy.deepcopy(run.metadata)
        metadata['protocol'] = common_protocol(run)
        normalized.append(replace(run, metadata=metadata))
    return original_validate(normalized)

def write(out_dir, runs, case_ids, analysis):
    analysis['comparison_contract'] = common_protocol(runs[0])
    analysis['method_protocols'] = {run.model: run.metadata['protocol'] for run in runs}
    analysis['protocol_note'] = 'Losses and decisions are method-specific; shared evaluation, data and ground-truth checks are enforced.'
    return original_write(out_dir, runs, case_ids, analysis)
if __name__ == '__main__':
    compare.validate_runs = validate
    compare.write_outputs = write
    raise SystemExit(compare.main())
