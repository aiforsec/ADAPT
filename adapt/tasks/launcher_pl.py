import subprocess


def get_all_commands():
    all_commands = []
    all_script_mapping = []

    # for _ds in ['androzoo-apigraph', 'bodmas', 'androzoo-drebin','ember-2018', 'pdf']:
    for _ds in ['androzoo-apigraph']:
        for _du in ['remove-intra']:
            for _m in ['svm']: 
            # for _m in ['xgboost', 'random-forest', 'mlp']:  
                for _s in range(1, 201):
                    # # HPO
                    all_commands.append('python -m deepmal.tasks.pseudo_labeling --source_free ' + '--dataset ' + _ds + ' --model '
                                        + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.baseline_insomia ' + '--dataset ' + _ds + ' --model '
                    #                      + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.baseline_morse ' + '--dataset ' + _ds + ' --model '
                    #                      + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.baseline_de ' + '--dataset ' + _ds + ' --model '
                    #                         + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.baseline_river ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.pseudo_labeling --source_free ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # TESTING
                    # all_commands.append('python -m deepmal.tasks.pseudo_labeling ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --test')
                    # all_commands.append('python -m deepmal.tasks.baseline_river ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --test')
                    # all_commands.append('python -m deepmal.tasks.baseline_de ' + '--dataset ' + _ds + ' --model '
                    # #                         + _m + ' --dupes ' + _du + ' --test')
                    # all_commands.append('python -m deepmal.tasks.baseline_insomia ' + '--dataset ' + _ds + ' --model '
                    #                      + _m + ' --dupes ' + _du + ' --test')
                    # all_commands.append('python -m deepmal.tasks.baseline_morse ' + '--dataset ' + _ds + ' --model '
                    #                      + _m + ' --dupes ' + _du + ' --test')
                    # source free
                    # all_commands.append('python -m deepmal.tasks.pseudo_labeling ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --test --source_free')

                    # # active learning
                    # all_commands.append('python -m deepmal.tasks.pseudo_labeling_active_learning ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.pseudo_labeling_active_learning ' + '--dataset ' + _ds + ' --model '
                    #                      + _m + ' --dupes ' + _du + ' --test ' + ' --test_seed ' + str(_s))

                    # just malware detection (no adaptation)
                    # all_commands.append('python -m deepmal.tasks.malware_detection ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --seed ' + str(_s))
                    # all_commands.append('python -m deepmal.tasks.malware_detection ' + '--dataset ' + _ds + ' --model '
                    #                     + _m + ' --dupes ' + _du + ' --test --skip_val')

                    
                    all_script_mapping.append('run_{}_hpo.sh'.format(_m))
    return all_commands, all_script_mapping


def slurm_launcher():
    all_commands, all_script_mapping = get_all_commands()
    print('\n'.join(all_commands))
    print('#commands:', len(all_commands))

    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        exit(0)
    # IMPLEMENT YOUR SLURM LAUNCHER with the commands


if __name__ == '__main__':
    slurm_launcher()
