output_dfa = {
	('-'): {	# negative first number
		('123456789'): {	
			('0123456789'): {	# two digits
				(' '): {
					('('): {
						('-'): {	# negative first expression number
							('123456789'): {
								('0123456789'): {	# two digits
									('+-*'): {
										('123456789'): {	# second expr op
											('0123456789'): {	# two digits
												(')'): '.'
											},
											(')'): '.'	# one digit	
										}
									}
								},
								('+-*'): {	# one digit
									('123456789'): {	# second expr op
										('0123456789'): {	# two digits
											(')'): '.'
										},
										(')'): '.'	# one digit	
									}
								}
							}
						},
						('123456789'): {	# positive first expression number
							('0123456789'): {	# two digits
								('+-*'): {
									('123456789'): {	# second expr op
										('0123456789'): {	# two digits
											(')'): '.'
										},
										(')'): '.'	# one digit	
									}
								}
							},
							('+-*'): {	# one digit
								('123456789'): {	# second expr op
									('0123456789'): {	# two digits
										(')'): '.'
									},
									(')'): '.'	# one digit	
								}
							}
						}
					}
				}
			},
			(' '): {	# one digit
				('('): {
					('-'): {	# negative first expr op
						('123456789'): {
							('0123456789'): {	# two digits
								('+-*'): {
									('123456789'): {	# second expr op
										('0123456789'): {	# two digits
											(')'): '.'
										},
										(')'): '.'	# one digit	
									}
								}
							},
							('+-*'): {	# one digit
								('123456789'): {	# second expr op
									('0123456789'): {	# two digits
										(')'): '.'
									},
									(')'): '.'	# one digit	
								}
							}
						}
					},
					('123456789'): {	# positive first expr op
						('0123456789'): {	# two digits
							('+-*'): {
								('123456789'): {	# second expr op
									('0123456789'): {	 # two digits
										(')'): '.'
									},
									(')'): '.'	# one digit	
								}
							}
						},
						('+-*'): {	# one digit
							('123456789'): {	# second expr op
								('0123456789'): {	# two digits
									(')'): '.'
								},
								(')'): '.'	# one digit		
							}
						}	
					}
				}
			}
		},
	},
	('123456789'): {	 # positive first number
		('0123456789'): {	# two digits
			(' '): {
				('('): {
					('-'): {	# negative first expr number
						('123456789'): {
							('0123456789'): {	# two digits
								('+-*'): {
									('123456789'): {
										('0123456789'): {	# two digits
											(')'): '.'
										},
										(')'): '.'	# one digit
									}
								}
							},
							('+-*'): {	# one digit
								('123456789'): {
									('0123456789'): {	# two digits
										(')'): '.'
									},
									(')'): '.'	# one digit
								}
							}	
						}
					},
					('123456789'): {	# positive first expr number
						('0123456789'): {	# two digits
							('+-*'): {
								('123456789'): {
									('0123456789'): {	# two digits
										(')'): '.'
									},
									(')'): '.'	# one digit
								}
							}
						},
						('+-*'): {
							('123456789'):  {	# two digits
								('0123456789'): {	# two digits
									(')'): '.'
								},
								(')'): '.'	# one digit
							}
						}	
					}
				}
			}
		},
		(' '): {	# one digit
			('('): {
				('-'): {	# negative first expression number
					('123456789'): {
						('0123456789'): {	# two digits
							('+-*'): {
								('123456789'): {
									('0123456789'): {	# two digits
										(')'): '.'
									},
									(')'): '.'	# one digit		
								}
							}
						},
						('+-*'): {	# one digit
							('123456789'): {
								('0123456789'): {	# two digits
									(')'): '.'
								},
								(')'): '.'	# one digit		
							}
						}	
					}
				},
				('123456789'): {	# positive first expression number
					('0123456789'): {	# two digits
						('+-*'): {
							('123456789'): {
								('0123456789'): {	# two digits
									(')'): '.'
								},
								(')'): '.'	# one digit		
							}
						}
					},
					('+-*'): {	# one digit
						('123456789'): {
							('0123456789'): {	# two digits
								(')'): '.'
							},
							(')'): '.'	# one digit		
						}
					}	
				}
			}
		}
	}
}