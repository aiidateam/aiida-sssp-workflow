def test_ld1_default(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_ld1):
    """Test ld1 parser
    """
    test_name = 'test_ld1_default'
    entry_point_calc_job = 'sssp.pseudo.ld1'
    entry_point_parser = 'sssp.pseudo.ld1'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, generate_inputs_ld1())
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_pseudo' in results
    