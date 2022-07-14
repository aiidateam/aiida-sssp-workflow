def test_oncv_default(fixture_localhost, generate_calc_job_node, generate_parser, data_regression):
    """Test oncv parser
    """
    name = 'default'
    entry_point_calc_job = 'sssp.pseudo.ld1'
    entry_point_parser = 'sssp.pseudo.ld1'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, True))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    assert 'output_pseudo' in results

    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict(),
    })
    